#include "soo_search_runtime.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>

#include <algorithm>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "soo/rules.hpp"

#ifdef DIAMOND_QT_HAS_SOO
#include "diamond_model/deployment_artifact.hpp"
#include "diamond_model/model_index.hpp"
#include "diamond_model/soo_evaluator.hpp"
#include "soo/mcts.hpp"
#include "soo/mcts3p.hpp"
#endif

#ifdef DIAMOND_QT_HAS_SOO
namespace {

// Packaged and source builds both name the promoted default in models/index.json.
// There is deliberately no spike/random fallback: a missing or malformed release
// model must fail loudly instead of producing apparently valid weak play.
QString resolve_artifact(const std::string& family) {
    const QStringList bases = {QCoreApplication::applicationDirPath(), QDir::currentPath()};
    for (const QString& base : bases) {
        const QString models = QDir(base).filePath(QStringLiteral("models"));
        if (!QFile::exists(QDir(models).filePath(QStringLiteral("index.json")))) continue;
        try {
            const auto index = diamond_model::load_model_index(models.toStdString());
            if (const auto* entry = index.default_for(family))
                return QString::fromStdString(entry->root.string());
        } catch (const std::exception&) {
            // A malformed index must not be a silent fallback to some other
            // model: let the artifact validation below fail loudly instead.
            return QString::fromStdString((std::filesystem::path(models.toStdString())).string());
        }
    }
    return QDir(QCoreApplication::applicationDirPath())
        .filePath(QStringLiteral("models"));
}

}  // namespace
#endif

class SooSearchRuntime::Impl {
  public:
    explicit Impl(QString artifactRoot) : artifact_root(std::move(artifactRoot)) {}

    QString artifact_root;
    std::mutex mutex;
#ifdef DIAMOND_QT_HAS_SOO
    std::unique_ptr<diamond_model::SooEvaluator> evaluator;
    diamond_model::DiamondModel min_model{nullptr};
    std::string loaded_family;

    void ensure_loaded(const std::string& family) {
        if (!loaded_family.empty()) {
            if (loaded_family != family)
                throw std::invalid_argument("Loaded model family does not match the game");
            return;
        }
        static std::once_flag configured;
        std::call_once(configured, [] {
            bool ok = false;
            const int requested = qEnvironmentVariableIntValue("DIAMOND_TORCH_THREADS", &ok);
            const int threads = ok && requested > 0 ? requested : 1;
            torch::set_num_threads(threads);
            torch::set_num_interop_threads(1);
        });
        const std::string root = (artifact_root.isEmpty() ? resolve_artifact(family)
                                                           : artifact_root).toStdString();
        const auto artifact = diamond_model::validate_deployment_artifact(root, family);
        diamond_model::DiamondModel model(artifact.width, artifact.residual_blocks,
                                          artifact.input_features, artifact.value_size);
        model->load_weights(artifact.weights);
        model->eval();
        if (family == "soo") evaluator = std::make_unique<diamond_model::SooEvaluator>(model);
        else min_model = std::move(model);
        loaded_family = family;
    }

    soo::SearchResult search_min(const soo::State& state, const soo::Match& match,
                                 const soo::MCTSConfig& config) {
        soo::SearchSession3P session(match, config);
        session.begin(state, 0.0);
        soo::SearchResult result;
        bool root = true;
        torch::NoGradGuard no_grad;
        while (session.advance() == soo::SearchSession3P::Status::NeedsEvaluation) {
            const auto started = std::chrono::steady_clock::now();
            const auto& encoded = session.pending_features();
            const auto& actions = session.pending_actions();
            auto features = torch::from_blob(const_cast<float*>(encoded.node_features.data()),
                                             {1, soo::kBoardSize, 6}, torch::kFloat32);
            const auto [logits, values] = min_model->forward(features);
            const auto indices = torch::tensor(actions, torch::TensorOptions().dtype(torch::kLong));
            const auto priors = torch::softmax(logits.index_select(1, indices), 1).contiguous();
            soo::EvalOutcome3P outcome;
            for (size_t index = 0; index < actions.size(); ++index) {
                const double prior = priors.data_ptr<float>()[index];
                if (!std::isfinite(prior)) throw std::runtime_error("Min produced a non-finite prior");
                outcome.priors.push_back(prior);
            }
            for (size_t index = 0; index < outcome.value.size(); ++index) {
                outcome.value[index] = values[0][static_cast<int64_t>(index)].item<float>();
                if (!std::isfinite(outcome.value[index]))
                    throw std::runtime_error("Min produced a non-finite value");
            }
            if (root) result.root_network_value = outcome.value[0];
            root = false;
            session.supply(outcome);
            result.neural_evaluation_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        }
        const auto& vector_result = session.result();
        result.selected_action = vector_result.selected_action;
        result.root_actions = vector_result.root_actions;
        result.visit_counts = vector_result.visit_counts;
        result.policy = vector_result.policy;
        result.root_priors = vector_result.root_priors;
        result.simulations_run = vector_result.simulations_run;
        result.evaluator_calls = vector_result.evaluator_calls;
        result.nodes_created = vector_result.nodes_created;
        const int seat = match.seat_of(state.current_player);
        uint64_t visits = 0;
        for (size_t index = 0; index < vector_result.q_vectors.size(); ++index) {
            const double q = vector_result.q_vectors[index].at(static_cast<size_t>(seat));
            result.q_values.push_back(q);
            result.root_mean_value += q * result.visit_counts[index];
            visits += result.visit_counts[index];
        }
        if (visits) result.root_mean_value /= static_cast<double>(visits);
        return result;
    }
#endif
};

SooSearchRuntime::SooSearchRuntime(QString artifactRoot)
    : impl_(std::make_unique<Impl>(std::move(artifactRoot))) {}
SooSearchRuntime::~SooSearchRuntime() = default;

AiSearchResult SooSearchRuntime::search(const soo::State& state, const soo::Match& match,
                                        const std::vector<int32_t>& rejected,
                                        int simulations) {
    std::scoped_lock lock(impl_->mutex);
#ifdef DIAMOND_QT_HAS_SOO
    if (match.count != 2 && match.count != 3)
        throw std::invalid_argument("Analysis requires a two-seat or three-seat match");
    impl_->ensure_loaded(match.count == 3 ? "min" : "soo");
    soo::MCTSConfig config;
    config.simulations = simulations;
    config.c_puct = 1.5;
    config.dirichlet_epsilon = 0.0;
    const auto started = std::chrono::steady_clock::now();
    const soo::SearchResult result = match.count == 3
        ? impl_->search_min(state, match, config)
        : soo::MCTS2P(match, *impl_->evaluator, config).run(state, 0.0, false);
    const double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();

    SearchTelemetry telemetry;
    telemetry.root_network_value = result.root_network_value;
    telemetry.root_search_value = result.root_mean_value;
    telemetry.total_ms = total_ms;
    telemetry.neural_ms = result.neural_evaluation_ms;
    telemetry.simulations = result.simulations_run;
    telemetry.evaluator_calls = result.evaluator_calls;
    telemetry.nodes_created = result.nodes_created;
    telemetry.actions.reserve(result.root_actions.size());
    for (size_t index = 0; index < result.root_actions.size(); ++index) {
        telemetry.actions.push_back(ActionTelemetry{
            soo::to_physical_action(result.root_actions[index], match, state.current_player),
            result.root_priors[index], result.q_values[index], result.visit_counts[index],
            result.policy[index]});
    }

    const auto is_rejected = [&rejected](int32_t action) {
        return std::find(rejected.cbegin(), rejected.cend(), action) != rejected.cend();
    };
    std::vector<size_t> ranked(result.root_actions.size());
    std::iota(ranked.begin(), ranked.end(), size_t{0});
    std::sort(ranked.begin(), ranked.end(), [&result](size_t left, size_t right) {
        if (result.visit_counts[left] != result.visit_counts[right])
            return result.visit_counts[left] > result.visit_counts[right];
        return result.root_actions[left] < result.root_actions[right];
    });
    for (size_t index : ranked) {
        const int32_t action = telemetry.actions[index].action;
        if (!is_rejected(action)) return AiSearchResult{action, std::move(telemetry)};
    }
    return AiSearchResult{
        soo::to_physical_action(result.selected_action, match, state.current_player),
        std::move(telemetry)};
#else
    std::vector<int32_t> legal;
    soo::legal_action_ids(state, legal);
    for (int32_t action : legal) {
        if (std::find(rejected.cbegin(), rejected.cend(), action) == rejected.cend())
            return AiSearchResult{action, {}};
    }
    return AiSearchResult{legal.empty() ? -1 : legal.front(), {}};
#endif
}
