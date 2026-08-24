#include "diamond_orchestration/rating.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

namespace diamond_orchestration {
namespace {

using Json = diamond_support::JsonValue;

template <size_t Count>
void validate_ids(const std::array<std::string, Count>& ids) {
    std::set<std::string> unique;
    for (const auto& id : ids) {
        if (id.empty()) throw RatingError("participant IDs must be non-empty");
        unique.insert(id);
    }
    if (unique.size() != Count) throw RatingError("participant IDs must be distinct");
}

template <size_t Count>
void validate_permutation(const std::array<int, Count>& values, const char* field) {
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    for (size_t i = 0; i < Count; ++i)
        if (sorted[i] != static_cast<int>(i + 1))
            throw RatingError(std::string(field) + " must be a physical-seat permutation");
}

template <size_t Count>
Json string_array(const std::array<std::string, Count>& values) {
    Json::Array result;
    for (const auto& value : values) result.emplace_back(value);
    return Json{std::move(result)};
}

template <size_t Count>
Json int_array(const std::array<int, Count>& values) {
    Json::Array result;
    for (int value : values) result.emplace_back(int64_t{value});
    return Json{std::move(result)};
}

Json nullable_string(const std::string& value) {
    return value.empty() ? Json{nullptr} : Json{value};
}

std::string event_digest(const char* event_type, Json::Object payload) {
    payload.emplace("event_type", Json{std::string(event_type)});
    return "sha256:" + diamond_support::sha256(diamond_support::canonical_json(Json{std::move(payload)}));
}

Json::Object soo_payload(const SooRatingEvent& event) {
    return {{"completed", Json{event.completed}},
            {"loser_id", nullable_string(event.loser_id)},
            {"opening_id", Json{event.opening_id}},
            {"participant_ids", string_array(event.participant_ids)},
            {"protocol_id", Json{event.protocol_id}},
            {"seat_assignment", int_array(event.seat_assignment)},
            {"sequence_index", Json{static_cast<int64_t>(event.sequence_index)}},
            {"turn_order", int_array(event.turn_order)},
            {"winner_id", nullable_string(event.winner_id)}};
}

Json::Object min_payload(const MinRatingEvent& event) {
    Json ranking{nullptr};
    if (event.completed) ranking = string_array(event.final_ranking);
    return {{"completed", Json{event.completed}},
            {"final_ranking", std::move(ranking)},
            {"opening_id", Json{event.opening_id}},
            {"participant_ids", string_array(event.participant_ids)},
            {"protocol_id", Json{event.protocol_id}},
            {"seat_assignment", int_array(event.seat_assignment)},
            {"sequence_index", Json{static_cast<int64_t>(event.sequence_index)}},
            {"turn_order", int_array(event.turn_order)}};
}

double normal_pdf(double value) {
    return (1.0 / std::sqrt(2.0 * std::acos(-1.0))) * std::exp(-0.5 * value * value);
}

// trueskill 0.4.5's default backend deliberately uses this Numerical Recipes
// erfc approximation rather than the platform libm implementation.
double trueskill_erfc(double value) {
    const double z = std::abs(value);
    const double t = 1.0 / (1.0 + z / 2.0);
    const double polynomial = 1.00002368 +
        t * (0.37409196 +
             t * (0.09678418 +
                  t * (-0.18628806 +
                       t * (0.27886807 +
                            t * (-1.13520398 +
                                 t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))));
    const double result = t * std::exp(-z * z - 1.26551223 + t * polynomial);
    return value < 0.0 ? 2.0 - result : result;
}

double normal_cdf(double value) { return 0.5 * trueskill_erfc(-value / std::sqrt(2.0)); }

struct Gaussian final {
    double pi = 0.0;
    double tau = 0.0;

    [[nodiscard]] double mu() const { return pi == 0.0 ? 0.0 : tau / pi; }
    [[nodiscard]] double sigma_squared() const {
        return pi == 0.0 ? std::numeric_limits<double>::infinity() : 1.0 / pi;
    }
};

Gaussian operator*(Gaussian left, Gaussian right) {
    return {left.pi + right.pi, left.tau + right.tau};
}

Gaussian operator/(Gaussian left, Gaussian right) {
    return {left.pi - right.pi, left.tau - right.tau};
}

struct Variable final {
    Gaussian value;
    std::map<const void*, Gaussian> messages;

    double update_value(const void* factor, Gaussian next) {
        const Gaussian previous = value;
        const Gaussian message = messages[factor];
        messages[factor] = (next * message) / previous;
        value = next;
        const double precision_delta = std::abs(value.pi - previous.pi);
        if (std::isinf(precision_delta)) return 0.0;
        return std::max(std::abs(value.tau - previous.tau), std::sqrt(precision_delta));
    }

    double update_message(const void* factor, Gaussian message) {
        const Gaussian previous = value;
        const Gaussian old = messages[factor];
        messages[factor] = message;
        value = (value / old) * message;
        const double precision_delta = std::abs(value.pi - previous.pi);
        if (std::isinf(precision_delta)) return 0.0;
        return std::max(std::abs(value.tau - previous.tau), std::sqrt(precision_delta));
    }

    [[nodiscard]] Gaussian without(const void* factor) const {
        const auto found = messages.find(factor);
        return found == messages.end() ? value : value / found->second;
    }
};

struct PriorFactor final {
    Variable& variable;
    double mu;
    double sigma;
    double dynamic;

    void down() const {
        const double next_sigma = std::sqrt(sigma * sigma + dynamic * dynamic);
        const double precision = std::pow(next_sigma, -2.0);
        (void)variable.update_value(this, {precision, precision * mu});
    }
};

struct LikelihoodFactor final {
    Variable& mean;
    Variable& value;
    double variance;

    void down() const {
        const Gaussian message = mean.without(this);
        const double a = 1.0 / (1.0 + variance * message.pi);
        (void)value.update_message(this, {a * message.pi, a * message.tau});
    }

    void up() const {
        const Gaussian message = value.without(this);
        const double a = 1.0 / (1.0 + variance * message.pi);
        (void)mean.update_message(this, {a * message.pi, a * message.tau});
    }
};

struct IdentityFactor final {
    Variable& mean;
    Variable& value;

    void down() const { (void)value.update_message(this, mean.without(this)); }
    void up() const { (void)mean.update_message(this, value.without(this)); }
};

struct SumFactor final {
    Variable& sum;
    std::array<Variable*, 2> terms;
    std::array<double, 2> coefficients;

    static Gaussian message_from(const std::array<Variable*, 2>& variables,
                                 const std::array<double, 2>& factors,
                                 const void* factor) {
        double inverse_precision = 0.0;
        double mean = 0.0;
        for (std::size_t index = 0; index < variables.size(); ++index) {
            const Gaussian value = variables[index]->without(factor);
            inverse_precision += factors[index] * factors[index] / value.pi;
            mean += factors[index] * value.mu();
        }
        const double precision = 1.0 / inverse_precision;
        return {precision, precision * mean};
    }

    void down() const { (void)sum.update_message(this, message_from(terms, coefficients, this)); }

    void up(std::size_t index) const {
        const std::size_t other = 1U - index;
        const std::array<Variable*, 2> variables{&sum, terms[other]};
        const std::array<double, 2> factors{
            1.0 / coefficients[index], -coefficients[other] / coefficients[index]};
        (void)terms[index]->update_message(this, message_from(variables, factors, this));
    }
};

double inverse_normal_cdf(double probability) {
    // Peter John Acklam's rational approximation, refined once by Newton's method.
    if (!(probability > 0.0 && probability < 1.0)) return probability == 0.0
        ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    constexpr std::array<double, 6> a = {-3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    constexpr std::array<double, 5> b = {-5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    constexpr std::array<double, 6> c = {-7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    constexpr std::array<double, 4> d = {7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};
    const auto polynomial = [](const auto& coefficients, double value) {
        double result = coefficients.front();
        for (std::size_t index = 1; index < coefficients.size(); ++index)
            result = result * value + coefficients[index];
        return result;
    };
    double result;
    if (probability < 0.02425) {
        const double q = std::sqrt(-2.0 * std::log(probability));
        result = polynomial(c, q) / (polynomial(d, q) * q + 1.0);
    } else if (probability > 1.0 - 0.02425) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - probability));
        result = -polynomial(c, q) / (polynomial(d, q) * q + 1.0);
    } else {
        const double q = probability - 0.5;
        const double r = q * q;
        result = polynomial(a, r) * q / (polynomial(b, r) * r + 1.0);
    }
    const double error = normal_cdf(result) - probability;
    return result - error / normal_pdf(result);
}

struct WinTruncateFactor final {
    Variable& variable;
    double draw_margin;

    double up() const {
        const Gaussian div = variable.without(this);
        const double c = std::sqrt(div.sigma_squared());
        const double difference = div.mu() / c;
        const double margin = draw_margin / c;
        const double cdf = std::max(normal_cdf(difference - margin), 1e-300);
        const double v = normal_pdf(difference - margin) / cdf;
        const double w = v * (v + difference - margin);
        const double denominator = 1.0 - w;
        return variable.update_value(this, {(div.pi / denominator),
                                             (div.tau + std::sqrt(div.pi) * v) / denominator});
    }
};

std::array<MinRating, 3> rate_ranked_min_event(const std::array<MinRating, 3>& ratings,
                                                const TrueSkillConfig& config) {
    std::array<Variable, 3> rating_variables;
    std::array<Variable, 3> performance_variables;
    std::array<Variable, 3> team_variables;
    std::array<Variable, 2> difference_variables;
    std::array<PriorFactor, 3> priors = {
        PriorFactor{rating_variables[0], ratings[0].mu, ratings[0].sigma, config.tau},
        PriorFactor{rating_variables[1], ratings[1].mu, ratings[1].sigma, config.tau},
        PriorFactor{rating_variables[2], ratings[2].mu, ratings[2].sigma, config.tau}};
    std::array<LikelihoodFactor, 3> likelihoods = {
        LikelihoodFactor{rating_variables[0], performance_variables[0], config.beta * config.beta},
        LikelihoodFactor{rating_variables[1], performance_variables[1], config.beta * config.beta},
        LikelihoodFactor{rating_variables[2], performance_variables[2], config.beta * config.beta}};
    std::array<IdentityFactor, 3> teams = {
        IdentityFactor{performance_variables[0], team_variables[0]},
        IdentityFactor{performance_variables[1], team_variables[1]},
        IdentityFactor{performance_variables[2], team_variables[2]}};
    std::array<SumFactor, 2> differences = {
        SumFactor{difference_variables[0], {&team_variables[0], &team_variables[1]}, {1.0, -1.0}},
        SumFactor{difference_variables[1], {&team_variables[1], &team_variables[2]}, {1.0, -1.0}}};
    const double draw_margin = config.draw_probability == 0.0 ? 0.0 :
        inverse_normal_cdf((config.draw_probability + 1.0) / 2.0) * std::sqrt(2.0) * config.beta;
    std::array<WinTruncateFactor, 2> truncations = {
        WinTruncateFactor{difference_variables[0], draw_margin},
        WinTruncateFactor{difference_variables[1], draw_margin}};

    for (const auto& factor : priors) factor.down();
    for (const auto& factor : likelihoods) factor.down();
    for (const auto& factor : teams) factor.down();
    for (std::size_t iteration = 0; iteration < 10; ++iteration) {
        differences[0].down();
        double delta = truncations[0].up();
        differences[0].up(1);
        differences[1].down();
        delta = std::max(delta, truncations[1].up());
        differences[1].up(0);
        if (delta <= 1e-4) break;
    }
    differences[0].up(0);
    differences[1].up(1);
    for (const auto& factor : teams) factor.up();
    for (const auto& factor : likelihoods) factor.up();

    std::array<MinRating, 3> result;
    for (std::size_t index = 0; index < result.size(); ++index)
        result[index] = {rating_variables[index].value.mu(),
                         std::sqrt(rating_variables[index].value.sigma_squared()), 0.0, 0};
    return result;
}

}  // namespace

void EloConfig::validate() const {
    if (!std::isfinite(initial_rating) || !std::isfinite(k_factor) || !std::isfinite(logistic_scale) ||
        k_factor <= 0 || logistic_scale <= 0 || rating_system_version != "soo-elo-v1")
        throw RatingError("invalid Soo Elo configuration");
}

void TrueSkillConfig::validate() const {
    if (!std::isfinite(mu) || !std::isfinite(sigma) || !std::isfinite(beta) || !std::isfinite(tau) ||
        !std::isfinite(draw_probability) || sigma <= 0 || beta <= 0 || tau < 0 ||
        draw_probability < 0 || draw_probability >= 1 || rating_system_version != "min-trueskill-v1")
        throw RatingError("invalid Min TrueSkill configuration");
}

void SooRatingEvent::validate() const {
    if (protocol_id.empty() || opening_id.empty()) throw RatingError("event protocol_id and opening_id must be non-empty");
    validate_ids(participant_ids); validate_permutation(seat_assignment, "seat_assignment"); validate_permutation(turn_order, "turn_order");
    if (completed) {
        if (winner_id.empty() || loser_id.empty() || winner_id == loser_id ||
            std::set<std::string>{winner_id, loser_id} != std::set<std::string>{participant_ids[0], participant_ids[1]})
            throw RatingError("completed Soo event requires a participant winner and loser");
    } else if (!winner_id.empty() || !loser_id.empty()) throw RatingError("aborted Soo event must not contain an outcome");
    if (event_id != event_digest("soo", soo_payload(*this))) throw RatingError("Soo event_id does not match event dimensions");
}

Json SooRatingEvent::to_json() const {
    validate();
    auto payload = soo_payload(*this);
    payload.emplace("event_id", Json{event_id});
    payload.emplace("event_type", Json{std::string("soo")});
    return Json{std::move(payload)};
}

void MinRatingEvent::validate() const {
    if (protocol_id.empty() || opening_id.empty()) throw RatingError("event protocol_id and opening_id must be non-empty");
    validate_ids(participant_ids); validate_permutation(seat_assignment, "seat_assignment"); validate_permutation(turn_order, "turn_order");
    if (completed) {
        validate_ids(final_ranking);
        if (std::set<std::string>(final_ranking.begin(), final_ranking.end()) !=
            std::set<std::string>(participant_ids.begin(), participant_ids.end()))
            throw RatingError("completed Min event requires a full participant ranking");
    } else if (std::any_of(final_ranking.begin(), final_ranking.end(), [](const auto& item) { return !item.empty(); }))
        throw RatingError("aborted Min event must not contain an outcome");
    if (event_id != event_digest("min", min_payload(*this))) throw RatingError("Min event_id does not match event dimensions");
}

Json MinRatingEvent::to_json() const {
    validate();
    auto payload = min_payload(*this);
    payload.emplace("event_id", Json{event_id});
    payload.emplace("event_type", Json{std::string("min")});
    return Json{std::move(payload)};
}

SooRatingEvent make_soo_rating_event(uint64_t sequence_index, std::string protocol_id,
                                     std::array<std::string, 2> participants, std::array<int, 2> seats,
                                     std::array<int, 2> order, std::string opening, bool completed,
                                     std::string winner, std::string loser) {
    SooRatingEvent event{sequence_index, std::move(protocol_id), std::move(participants), seats, order,
                         std::move(opening), completed, std::move(winner), std::move(loser), {}};
    event.event_id = event_digest("soo", soo_payload(event)); event.validate(); return event;
}

MinRatingEvent make_min_rating_event(uint64_t sequence_index, std::string protocol_id,
                                     std::array<std::string, 3> participants, std::array<int, 3> seats,
                                     std::array<int, 3> order, std::string opening, bool completed,
                                     std::array<std::string, 3> ranking) {
    MinRatingEvent event{sequence_index, std::move(protocol_id), std::move(participants), seats, order,
                         std::move(opening), completed, std::move(ranking), {}};
    event.event_id = event_digest("min", min_payload(event)); event.validate(); return event;
}

double expected_elo_score(double first, double second, const EloConfig& config) {
    config.validate();
    if (!std::isfinite(first) || !std::isfinite(second)) throw RatingError("Elo ratings must be finite");
    return 1.0 / (1.0 + std::pow(10.0, (second - first) / config.logistic_scale));
}

std::array<double, 2> rate_soo_match(double winner, double loser, bool completed, const EloConfig& config) {
    if (!completed) return {winner, loser};
    const double expected_winner = expected_elo_score(winner, loser, config);
    const double expected_loser = expected_elo_score(loser, winner, config);
    return {winner + config.k_factor * (1.0 - expected_winner), loser - config.k_factor * expected_loser};
}

RatingRegistry::RatingRegistry(std::string protocol_id, EloConfig config)
    : family_(Family::soo), protocol_id_(std::move(protocol_id)), elo_(std::move(config)) {
    if (protocol_id_.empty()) throw RatingError("rating protocol ID must be non-empty"); elo_.validate();
}

RatingRegistry::RatingRegistry(std::string protocol_id, TrueSkillConfig config)
    : family_(Family::min), protocol_id_(std::move(protocol_id)), trueskill_(std::move(config)) {
    if (protocol_id_.empty()) throw RatingError("rating protocol ID must be non-empty"); trueskill_.validate();
}

void RatingRegistry::add_participant(std::string id, std::string name) {
    if (id.empty() || name.empty()) throw RatingError("participant ID and display name must be non-empty");
    const auto [it, inserted] = participants_.emplace(std::move(id), std::move(name));
    if (!inserted) return;
    if (family_ == Family::soo) soo_ratings_[it->first] = elo_.initial_rating;
    else min_ratings_[it->first] = {trueskill_.mu, trueskill_.sigma, trueskill_.mu - 3.0 * trueskill_.sigma, 0};
}

bool RatingRegistry::record_event(const SooRatingEvent& event) {
    if (family_ != Family::soo) throw RatingError("Min registry accepts only Min events");
    event.validate();
    if (event.protocol_id != protocol_id_) throw RatingError("event protocol does not match registry protocol");
    for (const auto& id : event.participant_ids) if (!participants_.contains(id)) throw RatingError("event references unregistered participant");
    for (const auto& recorded : events_) if (const auto* current = std::get_if<SooRatingEvent>(&recorded); current && current->event_id == event.event_id) return false;
    events_.emplace_back(event); rebuild(); return true;
}

bool RatingRegistry::record_event(const MinRatingEvent& event) {
    if (family_ != Family::min) throw RatingError("Soo registry accepts only Soo events");
    event.validate();
    if (event.protocol_id != protocol_id_) throw RatingError("event protocol does not match registry protocol");
    for (const auto& id : event.participant_ids) if (!participants_.contains(id)) throw RatingError("event references unregistered participant");
    for (const auto& recorded : events_) if (const auto* current = std::get_if<MinRatingEvent>(&recorded); current && current->event_id == event.event_id) return false;
    events_.emplace_back(event); rebuild(); return true;
}

void RatingRegistry::rebuild() {
    if (family_ == Family::soo) {
        soo_ratings_.clear(); for (const auto& [id, ignored] : participants_) { (void)ignored; soo_ratings_[id] = elo_.initial_rating; }
        for (const auto& record : events_) if (const auto* event = std::get_if<SooRatingEvent>(&record); event->completed) {
            const auto updated = rate_soo_match(soo_ratings_.at(event->winner_id), soo_ratings_.at(event->loser_id), true, elo_);
            soo_ratings_[event->winner_id] = updated[0]; soo_ratings_[event->loser_id] = updated[1];
        }
        return;
    }
    min_ratings_.clear();
    for (const auto& [id, ignored] : participants_) { (void)ignored; min_ratings_[id] = {trueskill_.mu, trueskill_.sigma, trueskill_.mu - 3.0 * trueskill_.sigma, 0}; }
    for (const auto& record : events_) if (const auto* event = std::get_if<MinRatingEvent>(&record); event->completed) {
        const std::array<MinRating, 3> current = {
            min_ratings_.at(event->final_ranking[0]), min_ratings_.at(event->final_ranking[1]),
            min_ratings_.at(event->final_ranking[2])};
        const auto updated = rate_ranked_min_event(current, trueskill_);
        for (std::size_t index = 0; index < updated.size(); ++index) {
            auto& rating = min_ratings_.at(event->final_ranking[index]);
            rating = updated[index];
            rating.exposure = rating.mu - 3.0 * rating.sigma;
            ++rating.rated_games;
        }
    }
}

std::vector<SooLeaderboardEntry> RatingRegistry::soo_leaderboard() const {
    if (family_ != Family::soo) throw RatingError("not a Soo registry");
    std::map<std::string, uint64_t> games; for (const auto& [id, ignored] : participants_) { (void)ignored; games[id] = 0; }
    for (const auto& record : events_) if (const auto* event = std::get_if<SooRatingEvent>(&record); event->completed) for (const auto& id : event->participant_ids) ++games[id];
    std::vector<SooLeaderboardEntry> entries; for (const auto& [id, name] : participants_) entries.push_back({id, name, soo_ratings_.at(id), games[id]});
    std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) { return left.rating != right.rating ? left.rating > right.rating : left.participant_id < right.participant_id; }); return entries;
}

std::vector<MinLeaderboardEntry> RatingRegistry::min_leaderboard() const {
    if (family_ != Family::min) throw RatingError("not a Min registry");
    std::vector<MinLeaderboardEntry> entries; for (const auto& [id, name] : participants_) entries.push_back({id, name, min_ratings_.at(id)});
    std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) { return left.rating.exposure != right.rating.exposure ? left.rating.exposure > right.rating.exposure : left.participant_id < right.participant_id; }); return entries;
}

Json RatingRegistry::report_json() const {
    Json::Array rows;
    if (family_ == Family::soo) for (const auto& entry : soo_leaderboard()) rows.emplace_back(Json::Object{{"display_name", Json{entry.display_name}}, {"games", Json{static_cast<int64_t>(entry.games)}}, {"participant_id", Json{entry.participant_id}}, {"rating", Json{entry.rating}}});
    else for (const auto& entry : min_leaderboard()) rows.emplace_back(Json::Object{{"display_name", Json{entry.display_name}}, {"exposure", Json{entry.rating.exposure}}, {"mu", Json{entry.rating.mu}}, {"participant_id", Json{entry.participant_id}}, {"rated_games", Json{static_cast<int64_t>(entry.rating.rated_games)}}, {"sigma", Json{entry.rating.sigma}}});
    return Json{Json::Object{{"event_count", Json{static_cast<int64_t>(events_.size())}}, {"family", Json{family_ == Family::soo ? "soo" : "min"}}, {"leaderboard", Json{std::move(rows)}}, {"protocol_id", Json{protocol_id_}}}};
}

}  // namespace diamond_orchestration
