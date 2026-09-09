#include <filesystem>
#include <fstream>
#include <iostream>
#include "diamond_training/checkpoint_export.hpp"
#include "diamond_training/checkpoint.hpp"
#undef CHECK
#include "check.hpp"

int main(int argc, char** argv) {
    REQUIRE(argc == 2, "scratch path required");
    torch::set_num_threads(2);
    const std::filesystem::path scratch = argv[1];
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);
    auto model = diamond_model::DiamondModel(8, 1, 6, 3);
    model->set_adjacency(diamond_model::topology_adjacency());
    diamond_training::Trainer trainer(model,
        diamond_training::Compatibility::min("2.0.0", {1, 8}), {0.001, 0.0001},
        diamond_training::resolve_device("cpu"));
    const auto source = scratch / "checkpoint";
    diamond_training::CheckpointLineage lineage;
    lineage.run_id = "export-test";
    auto info = diamond_training::save_checkpoint_v3(source, trainer, lineage);
    const auto artifact = diamond_training::export_checkpoint(source, scratch / "artifact", "2.0.0-test");
    CHECK_EQ(artifact.model_family, "min");
    CHECK_EQ(artifact.width, 8);
    CHECK_EQ(artifact.model_sha256, diamond_training::canonical_model_digest(model));
    CHECK(std::filesystem::exists(info.generation / "optimizer.pt"));
    auto rejects = [](auto operation) { try { operation(); return false; } catch (const std::exception&) { return true; } };
    CHECK(rejects([&] { diamond_training::export_checkpoint(source, scratch / "artifact", "2.0.0-test"); }));
    trainer.model()->set_adjacency(torch::zeros({6, 73, 73}));
    diamond_training::save_checkpoint_v3(scratch / "zero", trainer, lineage);
    CHECK(rejects([&] { diamond_training::export_checkpoint(scratch / "zero", scratch / "bad", "2.0.0-test"); }));
    CHECK(!std::filesystem::exists(scratch / "bad"));
    const auto manifest_path = info.generation / "manifest.json";
    std::ifstream manifest_stream(manifest_path);
    const std::string original{std::istreambuf_iterator<char>(manifest_stream), {}};
    auto wrong_contract = original;
    const auto contract = wrong_contract.find("diamond-camp-relative-v1");
    REQUIRE(contract != std::string::npos, "encoder contract in manifest");
    wrong_contract.replace(contract, std::string("diamond-camp-relative-v1").size(), "unsupported-encoder-v1");
    std::ofstream(manifest_path) << wrong_contract;
    CHECK(rejects([&] { diamond_training::export_checkpoint(source, scratch / "wrong-contract", "2.0.0-test"); }));
    std::ofstream(manifest_path) << original;
    std::ofstream(info.generation / "state.pt", std::ios::trunc) << "corrupt";
    CHECK(rejects([&] { diamond_training::export_checkpoint(source, scratch / "corrupt", "2.0.0-test"); }));
    CHECK(!std::filesystem::exists(scratch / "corrupt"));
    std::filesystem::remove_all(scratch);
    return soo_test::report("checkpoint_export_test");
}
