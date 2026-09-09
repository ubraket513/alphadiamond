#include <QGuiApplication>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QElapsedTimer>
#include <QSettings>
#include <QTemporaryDir>
#include <QThread>
#include <QString>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "soo_search_runtime.hpp"
#include "native_controller.hpp"
#include "model_catalog.hpp"
#include "soo/encoder.hpp"
#include "soo/rules.hpp"
#include "diamond_model/deployment_artifact.hpp"
#include "diamond_model/soo_model.hpp"

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", QByteArrayLiteral("offscreen"));
    qputenv("DIAMOND_MCTS_SIMULATIONS", QByteArrayLiteral("4"));
    QGuiApplication application(argc, argv);
    if (argc != 2) return 2;
    try {
        soo::ensure_topology_configured();
        const auto artifact = diamond_model::validate_deployment_artifact(argv[1], "min");
        SooSearchRuntime runtime(QString::fromLocal8Bit(argv[1]));
        auto match = soo::standard_min_match();
        // Nonidentity seat order catches indexing utility by player ID instead of seat.
        std::swap(match.players[0], match.players[2]);
        soo::State state;
        for (int seat = 0; seat < match.count; ++seat)
            for (auto hole : soo::topology().camp_positions[match.players[seat].camp])
                state.occupancy[hole] = match.players[seat].id;
        diamond_model::DiamondModel model(artifact.width, artifact.residual_blocks,
                                          artifact.input_features, artifact.value_size);
        model->load_weights(artifact.weights);
        model->eval();
        torch::NoGradGuard no_grad;
        for (uint8_t player : {1, 2, 3}) {
            state.current_player = player;
            const auto result = runtime.search(state, match, {}, 4);
            std::vector<int32_t> legal;
            soo::legal_action_ids(state, legal);
            require(std::find(legal.begin(), legal.end(), result.selected_action) != legal.end(),
                    "Min search selected an illegal physical action");
            require(result.telemetry.simulations == 4 && result.telemetry.evaluator_calls > 1 &&
                        result.telemetry.nodes_created > 1 && result.telemetry.neural_ms > 0,
                    "Min move must run neural tree search, not the first-legal fallback");
            const auto encoded = soo::encode(state, match);
            auto features = torch::from_blob(const_cast<float*>(encoded.node_features.data()),
                                             {1, 73, 6}, torch::kFloat32);
            const auto [logits, values] = model->forward(features);
            require(std::abs(result.telemetry.root_network_value - values[0][0].item<float>()) < 1e-6,
                    "Min root value must belong to the canonical current player");
            double weighted_q = 0;
            uint32_t visits = 0;
            for (const auto& action : result.telemetry.actions) {
                require(std::find(legal.begin(), legal.end(), action.action) != legal.end(),
                        "Min telemetry uses canonical rather than physical actions");
                weighted_q += action.q * action.visits;
                visits += action.visits;
            }
            require(visits > 0 && std::abs(result.telemetry.root_search_value - weighted_q / visits) < 1e-9,
                    "Min root search value must be visit weighted");
            const auto retry = runtime.search(state, match, {result.selected_action}, 4);
            require(retry.selected_action != result.selected_action,
                    "Min Think Again must skip a rejected move");
            const auto one = runtime.search(state, match, {}, 1);
            const auto child = soo::apply_action(state, match, one.selected_action);
            const auto child_encoded = soo::encode(child, match);
            const auto component = std::find(child_encoded.canonical_player_ids.begin(),
                child_encoded.canonical_player_ids.end(), player) -
                child_encoded.canonical_player_ids.begin();
            const auto child_features = torch::from_blob(
                const_cast<float*>(child_encoded.node_features.data()), {1, 73, 6}, torch::kFloat32);
            const auto [child_logits, child_values] = model->forward(child_features);
            const auto selected = action_telemetry_for(one.telemetry, one.selected_action);
            require(selected && std::abs(selected->q - child_values[0][component].item<float>()) < 1e-6,
                    "Min Q must use the root player's component of the child value vector");
        }
        bool rejected = false;
        state.current_player = 1;
        try { (void)runtime.search(state, soo::standard_soo_match(), {}, 1); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "a loaded Min artifact must be refused for a Soo game");

        // Exercise the actual catalog -> controller -> worker -> committed telemetry flow.
        QTemporaryDir installed;
        require(installed.isValid(), "temporary model installation failed");
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, installed.path());
        const auto model_path = std::filesystem::path(installed.path().toStdString()) /
            "models" / "min" / artifact.model_version;
        std::filesystem::create_directories(model_path.parent_path());
        std::filesystem::copy(artifact.root, model_path, std::filesystem::copy_options::recursive);
        const QString alternate_version = QStringLiteral("2.0.0-min-restore-test.1");
        const auto alternate_path = model_path.parent_path() / alternate_version.toStdString();
        std::filesystem::copy(artifact.root, alternate_path, std::filesystem::copy_options::recursive);
        QFile alternate_metadata(QString::fromStdString((alternate_path / "metadata.json").string()));
        require(alternate_metadata.open(QIODevice::ReadOnly), "alternate metadata cannot be read");
        auto metadata = QJsonDocument::fromJson(alternate_metadata.readAll()).object();
        alternate_metadata.close();
        metadata["model_version"] = alternate_version;
        require(alternate_metadata.open(QIODevice::WriteOnly | QIODevice::Truncate),
                "alternate metadata cannot be written");
        alternate_metadata.write(QJsonDocument(metadata).toJson());
        alternate_metadata.close();
        diamond_model::validate_deployment_artifact(alternate_path, "min");
        const QString original_directory = QDir::currentPath();
        require(QDir::setCurrent(installed.path()), "temporary game directory failed");
        NativeController controller;
        qobject_cast<ModelCatalog*>(controller.modelCatalog())->selectModel(
            QStringLiteral("min/") + QString::fromStdString(artifact.model_version));
        require(controller.startMatch(QVariantList{3, 1, 2}, QVariantList{3}),
                "Min controller could not start");
        QElapsedTimer timer;
        timer.start();
        while (!controller.hasProposal() && timer.elapsed() < 30000) {
            application.processEvents();
            QThread::msleep(5);
        }
        require(controller.hasProposal() && controller.proposalIsAi(),
                "trained Min controller did not publish an AI move");
        require(controller.modelCatalog()->property("activeModelId").toString().startsWith(QStringLiteral("min/")) &&
                    controller.aiAgentName().contains(QStringLiteral("Min")),
                "Min game did not activate and name its own model family");
        require(controller.latestSearchCompute().value("simulations").toInt() == 4 &&
                    controller.latestSearchCompute().value("evaluatorCalls").toInt() > 1,
                "Min controller bypassed neural search");
        controller.confirmProposal();
        const auto rows = controller.positionTelemetry();
        require(rows.size() == 1 && rows.front().toMap().value("available").toBool() &&
                    rows.front().toMap().value("perspectivePlayerId").toInt() == 3,
                "Min committed telemetry must preserve the moving player's perspective");
        controller.setPerspectivePlayerId(2);
        require(controller.positionTelemetry() == rows,
                "Soo perspective selector must not invert Min placement values");
        const QUrl saved_game = QUrl::fromLocalFile(installed.filePath(QStringLiteral("min-game.json")));
        require(controller.saveGame(saved_game), "Min game could not be saved");
        controller.shutdown();
        const QString saved_id = QStringLiteral("min/") + QString::fromStdString(artifact.model_version);
        auto* catalog = qobject_cast<ModelCatalog*>(controller.modelCatalog());
        catalog->selectModel(QStringLiteral("min/") + alternate_version);
        require(controller.loadGame(saved_game) && catalog->activeModelId() == saved_id,
                "loading a Min save must restore its model instead of the current Min selection");
        controller.shutdown();
        NativeController startup;
        require(startup.playerCount() == 2 &&
                    !startup.modelCatalog()->property("activeModelId").toString().startsWith(QStringLiteral("min/")) &&
                    startup.aiAgentName().startsWith(QStringLiteral("Soo")),
                "default Soo startup must not activate the previously selected Min model");
        startup.shutdown();
        QFile saved_file(saved_game.toLocalFile());
        require(saved_file.open(QIODevice::ReadOnly), "saved Min game cannot be read");
        auto wrong_family = QJsonDocument::fromJson(saved_file.readAll()).object();
        saved_file.close();
        wrong_family["ai_model_id"] = QStringLiteral("soo/2.0.0");
        require(saved_file.open(QIODevice::WriteOnly | QIODevice::Truncate), "wrong-family save cannot be written");
        saved_file.write(QJsonDocument(wrong_family).toJson());
        saved_file.close();
        require(!controller.loadGame(saved_game) && controller.errorMessage().contains(QStringLiteral("family")),
                "a Min save naming a Soo model must be refused explicitly");
        QDir::setCurrent(original_directory);
    } catch (const std::exception& error) {
        qCritical("min_runtime_test: %s", error.what());
        return 1;
    }
    return 0;
}
