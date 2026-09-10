// Read-only CPU diagnostic using exactly the application's search implementation.
#include <QCoreApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "soo_search_runtime.hpp"
#include "soo/rules.hpp"

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    try {
        if (argc != 7)
            throw std::runtime_error("usage: audit GAME MODEL BUDGETS(comma-separated) SAMPLES OUT "
                                     "ORDER(actual|standard)");
        QFile input(QString::fromLocal8Bit(argv[1]));
        if (!input.open(QIODevice::ReadOnly))
            throw std::runtime_error("cannot read game");
        const auto game = QJsonDocument::fromJson(input.readAll()).object();
        if (game["status"].toString() != "finished")
            throw std::runtime_error("completed game required");
        const auto players = game["players"].toArray();
        if (players.size() != 3)
            throw std::runtime_error("Min game required");
        const QStringList camps{"x+", "y+", "z+", "x-", "y-", "z-"};
        soo::ensure_topology_configured();
        soo::Match match;
        match.count = 3;
        int ai = 0;
        for (int i = 0; i < 3; ++i) {
            const auto p = players[i].toObject();
            const int camp = camps.indexOf(p["camp"].toString());
            const int target = camps.indexOf(p["target_camp"].toString());
            if (camp < 0 || target < 0)
                throw std::runtime_error("invalid camp");
            match.players[i] = {static_cast<uint8_t>(p["id"].toInt()), static_cast<uint8_t>(camp),
                                static_cast<uint8_t>(target)};
            if (p["kind"].toString() == "ai")
                ai = p["id"].toInt();
        }
        if (!ai)
            throw std::runtime_error("AI seat missing");
        soo::State state;
        state.current_player = match.players[0].id;
        for (const auto& p : match.players)
            for (auto cell : soo::topology().camp_positions[p.camp])
                state.occupancy[cell] = p.id;
        struct Sample {
            soo::State state;
            int action;
            int previous_ai_action;
        };
        std::vector<Sample> samples;
        int previous = -1;
        for (const auto entry : game["history"].toArray()) {
            const auto row = entry.toObject();
            if (row["player_id"].toInt() != state.current_player)
                throw std::runtime_error("replay player mismatch");
            const auto move = row["move"].toObject();
            const int action = move["source"].toInt() * 73 + move["destination"].toInt();
            std::vector<int32_t> legal;
            soo::legal_action_ids(state, legal);
            if (std::find(legal.begin(), legal.end(), action) == legal.end())
                throw std::runtime_error("illegal saved action");
            if (state.current_player == ai) {
                samples.push_back({state, action, previous});
                previous = action;
            }
            state = soo::apply_action(state, match, action);
        }
        const auto occupancy = game["occupancy"].toArray();
        if (occupancy.size() != 73 || state.status != soo::kFinished)
            throw std::runtime_error("replay final state invalid");
        for (int i = 0; i < 73; ++i)
            if (occupancy[i].toInt() != state.occupancy[i])
                throw std::runtime_error("replay occupancy mismatch");
        int rank = -1;
        const auto finish = game["finish_order"].toArray();
        for (int i = 0; i < 3; ++i) {
            if (finish[i].toInt() != state.finish_order[i])
                throw std::runtime_error("replay finish mismatch");
            if (state.finish_order[i] == ai)
                rank = i;
        }
        const QString order = QString::fromLocal8Bit(argv[6]);
        if (order == "standard")
            std::sort(match.players.begin(), match.players.end(),
                      [](auto a, auto b) { return a.id < b.id; });
        else if (order != "actual")
            throw std::runtime_error("invalid order mode");
        const int count =
            std::min(QString::fromLocal8Bit(argv[4]).toInt(), static_cast<int>(samples.size()));
        if (count < 2)
            throw std::runtime_error("at least two samples required");
        SooSearchRuntime runtime(QString::fromLocal8Bit(argv[2]));
        QFile output(QString::fromLocal8Bit(argv[5]));
        if (output.exists() || !output.open(QIODevice::WriteOnly))
            throw std::runtime_error("output exists or unavailable");
        for (int n = 0; n < count; ++n) {
            const auto& sample =
                samples[static_cast<size_t>(n * (samples.size() - 1) / (count - 1))];
            for (const auto budgetText : QString::fromLocal8Bit(argv[3]).split(',')) {
                const int budget = budgetText.toInt();
                if (budget < 1 || budget > 16384)
                    throw std::runtime_error("invalid budget");
                const auto result = runtime.search(sample.state, match, {}, budget);
                const auto& t = result.telemetry;
                QJsonArray actions;
                for (const auto& a : t.actions)
                    actions.append(QJsonObject{{"action", a.action},
                                               {"prior", a.prior},
                                               {"q", a.q},
                                               {"visits", static_cast<int>(a.visits)}});
                std::vector<uint8_t> path;
                soo::canonical_move_path(sample.state, result.selected_action / 73,
                                         result.selected_action % 73, path);
                const QJsonObject row{{"ply", sample.state.turn_number},
                                      {"player", ai},
                                      {"model_path", QString::fromLocal8Bit(argv[2])},
                                      {"order", order},
                                      {"actual_outcome", 1 - rank},
                                      {"simulations", budget},
                                      {"nn", t.root_network_value},
                                      {"mcts", t.root_search_value},
                                      {"total_ms", t.total_ms},
                                      {"neural_ms", t.neural_ms},
                                      {"eval_calls", static_cast<int>(t.evaluator_calls)},
                                      {"selected_action", result.selected_action},
                                      {"recorded_action", sample.action},
                                      {"previous_ai_action", sample.previous_ai_action},
                                      {"selected_hops", static_cast<int>(path.size()) - 1},
                                      {"actions", actions}};
                output.write(QJsonDocument(row).toJson(QJsonDocument::Compact) + '\n');
                output.flush();
                std::cout << "ply=" << sample.state.turn_number << " sims=" << budget
                          << " nn=" << t.root_network_value << " mcts=" << t.root_search_value
                          << " ms=" << t.total_ms << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
