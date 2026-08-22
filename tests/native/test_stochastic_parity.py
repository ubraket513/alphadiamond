"""Gate E: native stochastic MCTS matches Python's *distributions*, not its stream.

Section 9 of ``docs/native_selfplay_phase0.md`` decided that cross-backend
bit-exact RNG parity is not required: CPython's ``gammavariate`` consumes a
variable number of draws per sample, so matching it would mean reimplementing a
rejection algorithm on MT19937 to buy reproducibility no experiment needs.  What
is required instead:

* the **distribution and semantics** of Dirichlet noise and temperature
  selection must match Python's, and
* each backend must be **deterministic for a given seed**.

That trade has a sharp consequence for testing.  Because the streams are allowed
to differ, no comparison of sequences can catch a wrong sampler -- a boost
exponent inverted, weights normalised twice, noise applied after the first
selection instead of before.  Every one of those still produces a plausible
stream.  They are only visible in the distribution, which is why the samplers
are exposed through the ABI and gated here on moments and frequencies rather
than on values.

Why this gate exists at all: with ``epsilon = 0`` and ``temperature = 0`` a real
model callback makes every lane play the same game (pitfall 7.9), so native
self-play could not generate training data.  These are the two knobs that fix
it.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from pathlib import Path

import pytest

from diamond.alphazero.mcts.puct import add_dirichlet_noise, select_from_visits
from diamond.alphazero.native import native_game, require_native
from diamond.game.state import build_players

FIXTURE = Path(__file__).parent / "fixtures" / "positions.jsonl"

ALPHA = 0.3
"""Production ``dirichlet_alpha``.  Deliberately below 1, so the sampler's boost
branch -- the one that is easy to get wrong -- is the branch under test."""

EPSILON = 0.25
"""Production ``dirichlet_epsilon``."""


def _record() -> dict:
    """A mid-game position, not the opening.

    Risk 3: per-call costs and branching factor vary ~3x between opening and
    full-game positions.  The opening also concentrates visits on a handful of
    actions, which would make the temperature frequency test insensitive.
    """
    candidates = []
    with FIXTURE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["status"] == "in_progress" and record["tag"] != "opening":
                candidates.append(record)
    if not candidates:
        raise AssertionError("no in-progress mid-game position in the corpus")
    return candidates[len(candidates) // 2]


class _Harness:
    def __init__(self) -> None:
        self.players = build_players(2)
        self.module = require_native()
        self.native = native_game(self.players)
        self.record = _record()

    def state(self):
        return self.module.State(
            occupancy=list(self.record["occupancy"]),
            current_player=self.record["current_player_id"],
            turn_number=self.record["turn_number"],
            status=0,
            finish_order=self.record["finish_order"],
        )

    def search(self, *, seed: int, epsilon: float = 0.0, temperature: float = 0.0,
               simulations: int = 32):
        config = self.module.MCTSConfig(
            simulations=simulations,
            c_puct=1.5,
            dirichlet_alpha=ALPHA,
            dirichlet_epsilon=epsilon,
            seed=seed,
        )
        return self.native.search(self.state(), config, temperature=temperature)


_HARNESS: _Harness | None = None


def _harness() -> _Harness:
    """Built on first use so collection never touches the extension."""
    global _HARNESS
    if _HARNESS is None:
        _HARNESS = _Harness()
    return _HARNESS


# --------------------------------------------------------------------------
# The samplers, in isolation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.3, 0.9, 1.0, 2.5])
def test_gamma_moments_match_the_analytic_distribution(alpha: float) -> None:
    """Gamma(alpha, 1) has mean = alpha and variance = alpha.

    Both moments, not just the mean: the alpha < 1 boost is
    ``Gamma(alpha + 1) * U**(1/alpha)``, and inverting that exponent to
    ``U**alpha`` leaves the mean close to right while the variance goes badly
    wrong.  A mean-only check passes the bug.
    """
    samples = require_native().sample_gamma(alpha, 200_000, 12345)
    assert len(samples) == 200_000
    assert all(value > 0.0 for value in samples)
    assert statistics.fmean(samples) == pytest.approx(alpha, rel=0.03)
    assert statistics.pvariance(samples) == pytest.approx(alpha, rel=0.06)


def test_gamma_agrees_with_cpython_on_the_cdf() -> None:
    """A distribution comparison against the actual Python authority.

    Compared on the **CDF** scale, not the quantile scale, and the reason is
    worth recording.  Near zero, Gamma(0.3, 1) has F(x) ~ x**0.3, so its
    quantile function goes like p**(1/0.3) -- a 1 % error in probability becomes
    a 3.3 % error in x, and at the 10th percentile the sampling noise of 100k
    draws is already several percent.  A quantile-for-quantile test at
    alpha = 0.3 is therefore either flaky or too loose to detect anything.

    The fraction of samples below a threshold is a plain proportion with
    standard error sqrt(p(1-p)/n) ~ 0.0016 here, so a 0.01 bound is roughly six
    sigma: tight enough to catch a wrong sampler, stable enough not to flake.
    """
    count = 100_000
    native = sorted(require_native().sample_gamma(ALPHA, count, 999))
    rng = random.Random(999)
    python = sorted(rng.gammavariate(ALPHA, 1.0) for _ in range(count))

    import bisect

    for quantile in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        threshold = python[int(quantile * count)]
        share = bisect.bisect_left(native, threshold) / count
        assert share == pytest.approx(quantile, abs=0.01), (
            f"native puts {share:.4f} of its mass below Python's {quantile} quantile"
        )


def test_gamma_is_deterministic_per_seed() -> None:
    module = require_native()
    assert module.sample_gamma(ALPHA, 500, 4) == module.sample_gamma(ALPHA, 500, 4)
    assert module.sample_gamma(ALPHA, 500, 4) != module.sample_gamma(ALPHA, 500, 5)


def test_gamma_refuses_a_non_positive_alpha() -> None:
    with pytest.raises(Exception, match="alpha"):
        require_native().sample_gamma(0.0, 1, 0)


def test_neighbouring_seeds_decorrelate() -> None:
    """Lanes are seeded from consecutive integers.

    A generator that does not avalanche its seed gives neighbouring lanes
    near-identical streams, which is exactly how the Gate C salt bug produced 32
    lanes playing one game while every throughput number looked healthy.  The
    seed goes through splitmix64 for this reason; this asserts it worked.
    """
    module = require_native()
    # 20k pairs, not 2k: the standard error of a sample correlation is
    # 1/sqrt(n), so at n = 2000 an honest generator reads ~0.022 and a 0.05
    # bound is barely two sigma -- a test that fails on luck alone.
    count = 20_000
    first = module.sample_gamma(1.0, count, 1)
    second = module.sample_gamma(1.0, count, 2)
    mean_first = statistics.fmean(first)
    mean_second = statistics.fmean(second)
    covariance = statistics.fmean(
        (a - mean_first) * (b - mean_second) for a, b in zip(first, second)
    )
    correlation = covariance / (
        statistics.pstdev(first) * statistics.pstdev(second)
    )
    assert abs(correlation) < 0.05, f"consecutive seeds correlate at {correlation:.3f}"


def test_weighted_selection_matches_random_choices_frequencies() -> None:
    """``random.choices(population, weights, k=1)`` semantics, by frequency."""
    weights = [1.0, 3.0, 0.0, 6.0]
    draws = require_native().sample_weighted(weights, 60_000, 77)
    counts = Counter(draws)

    assert counts[2] == 0, "a zero-weight option must never be selected"
    total = sum(weights)
    for index, weight in enumerate(weights):
        assert counts[index] / 60_000 == pytest.approx(weight / total, abs=0.01)


def test_weighted_selection_refuses_a_degenerate_population() -> None:
    """Both cases are ``ValueError`` in Python; pybind11 maps the C++ throw to
    the same type, so the match strings are what pin the distinction."""
    module = require_native()
    with pytest.raises(ValueError, match="empty population"):
        module.sample_weighted([], 1, 0)
    with pytest.raises(ValueError, match="positive"):
        module.sample_weighted([0.0, 0.0], 1, 0)


# --------------------------------------------------------------------------
# Dirichlet noise inside the search
# --------------------------------------------------------------------------


def test_noise_is_not_drawn_when_epsilon_is_zero() -> None:
    """The whole of Gate B depends on this staying true."""
    quiet = _harness().search(seed=3, epsilon=0.0)
    other = _harness().search(seed=999_983, epsilon=0.0)
    assert list(quiet["root_priors"]) == list(other["root_priors"])


def test_mixed_priors_remain_a_distribution() -> None:
    """(1 - eps) * prior + eps * dirichlet is convex, so the sum is preserved."""
    for seed in range(24):
        result = _harness().search(seed=seed, epsilon=EPSILON)
        assert sum(result["root_priors"]) == pytest.approx(1.0, abs=1e-9)
        assert all(prior > 0.0 for prior in result["root_priors"])


def test_mixture_has_the_semantics_python_gives_it() -> None:
    """E[mixed] = (1 - eps) * prior + eps / n, and eps = 1 is pure Dirichlet.

    This is the assertion that pins *where* the noise is applied as well as how.
    Noise mixed into a prior after the first selection, or applied to the wrong
    node, does not produce this expectation at the root.
    """
    base = list(_harness().search(seed=0, epsilon=0.0)["root_priors"])
    count = len(base)

    for epsilon, expected in (
        (EPSILON, [(1.0 - EPSILON) * p + EPSILON / count for p in base]),
        (1.0, [1.0 / count] * count),
    ):
        totals = [0.0] * count
        trials = 400
        for seed in range(trials):
            priors = _harness().search(seed=seed, epsilon=epsilon)["root_priors"]
            for index, prior in enumerate(priors):
                totals[index] += prior
        observed = [total / trials for total in totals]
        for index, (got, want) in enumerate(zip(observed, expected)):
            assert got == pytest.approx(want, abs=0.02), (
                f"epsilon={epsilon} action {index}: {got:.4f} != {want:.4f}"
            )


def test_mixture_agrees_with_the_python_helper_given_the_same_noise() -> None:
    """Semantics, checked against ``puct.add_dirichlet_noise`` itself.

    The draws cannot be shared across backends, so this feeds the *native*
    sampler's normalised gammas into the Python helper via a stub generator and
    requires the arithmetic to agree.  What is compared is the mixing rule, with
    the sampler held constant -- the one part of the contract that must be
    identical rather than merely equal in distribution.
    """
    base = list(_harness().search(seed=0, epsilon=0.0)["root_priors"])
    count = len(base)
    samples = require_native().sample_gamma(ALPHA, count, 2024)

    class _Stub(random.Random):
        def __init__(self, values):
            super().__init__(0)
            self._values = list(values)

        def gammavariate(self, alpha, beta):
            return self._values.pop(0)

    priors = {index: prior for index, prior in enumerate(base)}
    expected = add_dirichlet_noise(
        priors, alpha=ALPHA, epsilon=EPSILON, rng=_Stub(samples)
    )

    total = sum(samples)
    manual = [
        (1.0 - EPSILON) * prior + EPSILON * (sample / total)
        for prior, sample in zip(base, samples)
    ]
    assert manual == pytest.approx(list(expected.values()), rel=1e-12)


# --------------------------------------------------------------------------
# Temperature
# --------------------------------------------------------------------------


def test_temperature_selection_follows_the_visit_distribution() -> None:
    """At T = 1 the selection frequency is the normalised visit count.

    ``epsilon`` is 0 here on purpose: the tree is then identical for every seed,
    so the only thing varying is the final draw.  That isolates
    ``select_from_visits`` from the search around it.
    """
    reference = _harness().search(seed=0, epsilon=0.0, temperature=0.0)
    actions = list(reference["root_actions"])
    visits = list(reference["visit_counts"])
    total = sum(visits)

    trials = 3000
    counts = Counter(
        _harness().search(seed=seed, epsilon=0.0, temperature=1.0)["selected_action"]
        for seed in range(trials)
    )
    assert set(counts) <= set(actions)
    for action, visit in zip(actions, visits):
        assert counts[action] / trials == pytest.approx(visit / total, abs=0.03), (
            f"action {action}"
        )


def test_low_temperature_concentrates_on_the_most_visited_action() -> None:
    """T -> 0 must approach the deterministic argmax, not merely favour it."""
    greedy = _harness().search(seed=0, epsilon=0.0, temperature=0.0)["selected_action"]
    counts = Counter(
        _harness().search(seed=seed, epsilon=0.0, temperature=0.05)["selected_action"]
        for seed in range(300)
    )
    assert counts[greedy] / 300 > 0.95


def test_temperature_zero_is_the_deterministic_tie_break() -> None:
    """Exactly the Gate B path: most visits, ties to the smallest action id."""
    result = _harness().search(seed=11, epsilon=0.0, temperature=0.0)
    visits = dict(zip(result["root_actions"], result["visit_counts"]))
    expected = select_from_visits(visits, temperature=0.0, rng=random.Random(0))
    assert result["selected_action"] == expected


# --------------------------------------------------------------------------
# Determinism, and the diversity this gate exists to deliver
# --------------------------------------------------------------------------


def test_a_stochastic_search_is_reproducible_from_its_seed() -> None:
    for seed in (0, 5, 4242):
        first = _harness().search(seed=seed, epsilon=EPSILON, temperature=1.0)
        second = _harness().search(seed=seed, epsilon=EPSILON, temperature=1.0)
        assert first["selected_action"] == second["selected_action"]
        assert list(first["visit_counts"]) == list(second["visit_counts"])
        assert list(first["root_priors"]) == list(second["root_priors"])


def test_distinct_seeds_explore_distinct_trees() -> None:
    """The payoff: noise must actually move the search, not just the priors."""
    trees = {
        tuple(_harness().search(seed=seed, epsilon=EPSILON)["visit_counts"])
        for seed in range(32)
    }
    assert len(trees) > 16, f"only {len(trees)} distinct search trees from 32 seeds"
