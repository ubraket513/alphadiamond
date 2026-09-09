#include "diamond_training/checkpoint_export.hpp"
#include "diamond_training/checkpoint.hpp"
#include "diamond_support/json.hpp"
#include "soo/board.hpp"
#include "soo/encoder.hpp"
#include "soo/rules.hpp"
#include <bit>
#include <fstream>
#include <iterator>

namespace diamond_training {
namespace {
using Json = diamond_support::JsonValue;
using Object = Json::Object;
std::string read(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw CheckpointError("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(stream), {}};
}
void write(const std::filesystem::path& path, const void* data, size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream) throw CheckpointError("cannot write " + path.string());
}
template<class Table> void write_i32(const std::filesystem::path& path, const Table& table) {
    std::vector<int32_t> values;
    for (const auto& row : table) for (const auto value : row) values.push_back(value);
    write(path, values.data(), values.size() * sizeof(int32_t));
}
void parity(diamond_model::DiamondModel& source, diamond_model::DiamondModel& deployed, bool min) {
    torch::NoGradGuard guard;
    source->eval();
    deployed->eval();
    const auto match = min ? soo::standard_min_match() : soo::standard_soo_match();
    soo::State state;
    state.current_player = match.players[0].id;
    for (int seat = 0; seat < match.count; ++seat)
        for (auto hole : soo::topology().camp_positions[match.players[seat].camp])
            state.occupancy[hole] = match.players[seat].id;
    for (int ply = 0; ply < 12; ++ply) {
        auto encoded = soo::encode(state, match);
        auto features = torch::from_blob(encoded.node_features.data(),
            {1, 73, source->input_features()}).clone();
        const auto [policy, value] = source->forward(features);
        const auto [other_policy, other_value] = deployed->forward(features);
        if (!torch::equal(policy, other_policy) || !torch::equal(value, other_value) ||
            !torch::isfinite(policy).all().item<bool>() || !torch::isfinite(value).all().item<bool>())
            throw CheckpointError("deployment CPU inference parity failed");
        std::vector<int32_t> actions;
        soo::legal_action_ids(state, actions);
        if (actions.empty()) throw CheckpointError("parity position has no legal actions");
        state = soo::apply_action(state, match, actions.at((ply * 7) % actions.size()));
    }
}
}

diamond_model::DeploymentArtifact export_checkpoint(const std::filesystem::path& checkpoint,
    const std::filesystem::path& destination, const std::string& version) {
    if constexpr (std::endian::native != std::endian::little)
        throw CheckpointError("raw deployment export requires a little-endian host");
    if (version.empty()) throw CheckpointError("deployment version must not be empty");
    if (std::filesystem::exists(destination)) throw CheckpointError("deployment destination already exists");
    const auto info = validate_checkpoint_v2(checkpoint);
    if (info.format_version != 3 || !info.provenance)
        throw CheckpointError("deployment export requires native v3 architecture provenance");
    const auto manifest = std::get<Object>(diamond_support::parse_json(read(info.generation / "manifest.json")).value);
    const auto provenance = std::get<Object>(manifest.at("provenance").value);
    const auto architecture = std::get<Object>(provenance.at("architecture").value);
    auto dimension = [&](const char* name) { return std::get<int64_t>(architecture.at(name).value); };
    const auto width = dimension("width"), blocks = dimension("residual_blocks");
    const auto features = dimension("input_features"), values = dimension("value_size");
    const bool min = features == 6 && values == 3;
    if (!min && !(features == 4 && values == 1)) throw CheckpointError("unsupported deployment model family");
    const auto compatibility = std::get<Object>(provenance.at("compatibility").value);
    if (std::get<std::string>(compatibility.at("model_name").value) != (min ? "Min" : "Soo"))
        throw CheckpointError("checkpoint family and architecture disagree");
    const auto expected = min ? Compatibility::min(version, {blocks, width})
                              : Compatibility::soo(version, {blocks, width});
    for (const auto& [name, value] : std::map<std::string, std::string>{
        {"board_topology_version", expected.board_topology_version},
        {"encoder_version", expected.encoder_version},
        {"action_space_version", expected.action_space_version},
        {"ruleset_version", expected.ruleset_version},
        {"ruleset_fingerprint", expected.ruleset_fingerprint},
        {"seat_layout_version", expected.seat_layout_version},
        {"value_semantics_version", expected.value_semantics_version}})
        if (std::get<std::string>(compatibility.at(name).value) != value)
            throw CheckpointError("unsupported checkpoint contract: " + name);
    const auto network = std::get<Object>(compatibility.at("network_config").value);
    if (std::get<int64_t>(network.at("width").value) != width ||
        std::get<int64_t>(network.at("residual_blocks").value) != blocks ||
        std::get<int64_t>(compatibility.at("player_count").value) != expected.player_count)
        throw CheckpointError("checkpoint compatibility and architecture disagree");
    auto model = diamond_model::DiamondModel(width, blocks, features, values);
    const auto loaded = load_checkpoint_v2_weights(checkpoint, model, resolve_device("cpu"));
    if (loaded.generation != info.generation || loaded.model_digest != info.model_digest)
        throw CheckpointError("checkpoint model identity mismatch");
    if (!torch::equal(model->adjacency, diamond_model::topology_adjacency()))
        throw CheckpointError("checkpoint adjacency is not the canonical topology; refusing export");
    for (const auto& parameter : model->parameters())
        if (!torch::isfinite(parameter).all().item<bool>()) throw CheckpointError("checkpoint contains non-finite weights");
    // Reserve the destination exclusively. On any failure, remove only this new directory.
    if (destination.has_parent_path()) std::filesystem::create_directories(destination.parent_path());
    if (!std::filesystem::create_directory(destination)) throw CheckpointError("deployment destination already exists");
    try {
        std::filesystem::create_directory(destination / "weights");
        auto tensor = [&](const std::string& name, const torch::Tensor& value) {
            auto cpu = value.detach().to(torch::kCPU).contiguous();
            if (cpu.scalar_type() != torch::kFloat32) throw CheckpointError("deployment weights must be float32");
            write(destination / "weights" / (name + ".f32"), cpu.const_data_ptr<float>(), cpu.numel() * sizeof(float));
        };
        auto module = [&](const std::string& name, const auto& value) {
            for (const auto& parameter : value->named_parameters(false)) tensor(name + "__" + parameter.key(), parameter.value());
        };
        module("trunk__input_projection", model->input_projection);
        for (int64_t i = 0; i < blocks; ++i) {
            const auto prefix = "trunk__blocks__" + std::to_string(i) + "__";
            auto block = model->blocks.at(i);
            module(prefix + "self_projection", block->self_projection);
            for (int d = 0; d < 6; ++d) module(prefix + "direction_projections__" + std::to_string(d), block->direction_projections.at(d));
            module(prefix + "norm", block->norm);
        }
        module("trunk__output_norm", model->output_norm);
        module("policy_head__source", model->policy_source);
        module("policy_head__destination", model->policy_destination);
        module("value_head__0", model->value_linear1);
        module("value_head__2", model->value_linear2);
        tensor("trunk__adjacency", model->adjacency);
        const auto topology = soo::generate_topology();
        write(destination / "topology_neighbour.i8", topology.neighbour.data(), 73 * 6);
        write_i32(destination / "topology_camp_positions.i32", topology.camp_positions);
        write_i32(destination / "topology_pairwise_distance.i32", topology.pairwise);
        write_i32(destination / "topology_physical_to_canonical.i32", topology.physical_to_canonical);
        write_i32(destination / "topology_canonical_to_physical.i32", topology.canonical_to_physical);
        Json commit{nullptr};
        if (info.provenance->source_git_commit != "unavailable") commit = Json{info.provenance->source_git_commit};
        const auto metadata = diamond_support::canonical_json(Json{Object{
            {"architecture", Json{Object{{"residual_blocks", Json{blocks}}, {"width", Json{width}}, {"type", Json{std::string("directional_residual")}}}}},
            {"corpus_seed", Json{int64_t{0}}}, {"dtype", Json{std::string("float32")}}, {"format_version", Json{int64_t{3}}},
            {"game_contract", Json{Object{{"action_space", Json{std::string("diamond73-srcdst-v1")}}, {"encoder", Json{std::string("diamond-camp-relative-v1")}}, {"topology", Json{std::string("diamond73-v1")}}}}},
            {"model_family", Json{std::string(min ? "min" : "soo")}}, {"model_version", Json{version}},
            {"model_sha256", Json{canonical_model_digest(model)}},
            {"runtime_sha256", Json{diamond_model::deployment_runtime_sha256(destination, width, blocks, features, values)}},
            {"source", Json{Object{{"checkpoint_sha256", Json{info.model_digest}}, {"training_commit", commit}, {"training_step", Json{static_cast<int64_t>(info.training_step)}}}}},
            {"tensor_shapes", Json{Object{{"input", Json{Json::Array{Json{int64_t{1}}, Json{int64_t{73}}, Json{features}}}}, {"policy", Json{Json::Array{Json{int64_t{1}}, Json{int64_t{5329}}}}}, {"value", Json{Json::Array{Json{int64_t{1}}, Json{values}}}}}}}
        }});
        write(destination / "metadata.json", metadata.data(), metadata.size());
        auto artifact = diamond_model::validate_deployment_artifact(destination);
        auto deployed = diamond_model::DiamondModel(width, blocks, features, values);
        deployed->load_weights(artifact.weights);
        parity(model, deployed, min);
        return artifact;
    } catch (...) {
        std::filesystem::remove_all(destination);
        throw;
    }
}
}
