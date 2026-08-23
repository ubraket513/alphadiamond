#include "soo_search_runtime.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>

#include <algorithm>
#include <filesystem>
#include <chrono>
#include <mutex>
#include <numeric>
#include <stdexcept>

#include "soo/rules.hpp"

#ifdef DIAMOND_QT_HAS_SOO
#include "diamond_model/deployment_artifact.hpp"
#include "diamond_model/model_index.hpp"
#include "diamond_model/soo_evaluator.hpp"
#include "soo/mcts.hpp"
#endif

#ifdef DIAMOND_QT_HAS_SOO
namespace {

// A packaged application ships its models beside the executable and names the
// default in models/index.json; a development tree has artifacts/soo-spike.
// Try the package first so a release never silently picks up a stale spike.
QString resolve_soo_artifact() {
    const QStringList bases = {QCoreApplication::applicationDirPath(), QDir::currentPath()};
    for (const QString& base : bases) {
        const QString models = QDir(base).filePath(QStringLiteral("models"));
        if (!QFile::exists(QDir(models).filePath(QStringLiteral("index.json")))) continue;
        try {
            const auto index = diamond_model::load_model_index(models.toStdString());
            if (const auto* entry = index.default_for("soo"))
                return QString::fromStdString(entry->root.string());
        } catch (const std::exception&) {
            // A malformed index must not be a silent fallback to some other
            // model: let the artifact validation below fail loudly instead.
            return QString::fromStdString((std::filesystem::path(models.toStdString())).string());
        }
    }
    for (const QString& base : bases) {
        const QString spike = QDir(base).filePath(QStringLiteral("artifacts/soo-spike"));
        if (QDir(spike).exists()) return spike;
    }
    return QDir(QCoreApplication::applicationDirPath())
        .filePath(QStringLiteral("artifacts/soo-spike"));
}

}  // namespace
#endif

class SooSearchRuntime::Impl {
  public:
    std::mutex mutex;
#ifdef DIAMOND_QT_HAS_SOO
    std::unique_ptr<diamond_model::SooEvaluator> evaluator;

    void ensure_loaded() {
        if (evaluator) return;
        static std::once_flag configured;
        std::call_once(configured, [] {
            bool ok = false;
            const int requested = qEnvironmentVariableIntValue("DIAMOND_TORCH_THREADS", &ok);
            const int threads = ok && requested > 0 ? requested : 1;
            torch::set_num_threads(threads);
            torch::set_num_interop_threads(1);
        });
        const std::string root = resolve_soo_artifact().toStdString();
        // Family-scoped: a Min bundle in the Soo slot is refused rather than
        // loaded with the wrong tensor shapes.
        const auto artifact = diamond_model::validate_deployment_artifact(root, "soo");
        diamond_model::DiamondModel model(artifact.width, artifact.residual_blocks,
                                          artifact.input_features, artifact.value_size);
        model->load_weights(artifact.weights);
        evaluator = std::make_unique<diamond_model::SooEvaluator>(model);
    }
#endif
};

SooSearchRuntime::SooSearchRuntime() : impl_(std::make_unique<Impl>()) {}
SooSearchRuntime::~SooSearchRuntime() = default;

AiSearchResult SooSearchRuntime::search(const soo::State& state, const soo::Match& match,
                                        const std::vector<int32_t>& rejected,
                                        int simulations) {
    std::scoped_lock lock(impl_->mutex);
#ifdef DIAMOND_QT_HAS_SOO
    if (match.count != 2) throw std::invalid_argument("Soo analysis requires a two-seat match");
    impl_->ensure_loaded();
    soo::MCTSConfig config;
    config.simulations = simulations;
    config.c_puct = 1.5;
    config.dirichlet_epsilon = 0.0;
    soo::MCTS2P search(match, *impl_->evaluator, config);
    const auto started = std::chrono::steady_clock::now();
    const soo::SearchResult result = search.run(state, 0.0, false);
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

