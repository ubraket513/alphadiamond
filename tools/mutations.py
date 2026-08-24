"""What to break, and which gate must notice.

Each entry is `(file, before, after, gate)`. `tools/mutation_check.py` applies
them one at a time to the real source, builds, runs the native lane, and
requires the named gate to fail.

The set is chosen to be *behavioural*: a mutation that changes no observable
behaviour proves nothing when it goes unnoticed. It is also chosen to cover the
mistakes each gate exists for -- a wrong player channel, a rotated seat mapping,
a negated vector, an inverted tie-break -- rather than arbitrary damage, because
a gate that only catches nonsense is not evidence that it would catch a port
bug.
"""

MUTATIONS = [
    (
        "native/src/rules.cpp",
        "int moves_from(const State& state, int source, uint8_t* dest_out, uint8_t* kind_out) {",
        (
            "int moves_from(const State& state, int source, uint8_t* dest_out,"
            " uint8_t* kind_out) {\n"
            "    if (source == 41) return 0;  // MUTATION: drop one source's moves"
        ),
        "rules_golden_test",
    ),
    (
        # The occupancy goes into the wrong player channel: the network sees a
        # coherent board with the seats swapped.
        "native/src/encoder.cpp",
        "            row[channel] = 1.0f;",
        "            row[(channel + 1) % players] = 1.0f;  // MUTATION: wrong channel",
        "rules_golden_test",
    ),
    (
        # The canonical rotation: encode from the wrong camp's mapping.
        "native/src/encoder.cpp",
        "    const auto& mapping = topo.physical_to_canonical[home_camp];",
        (
            "    const auto& mapping = topo.physical_to_canonical[(home_camp + 1) % kCamps];"
            "  // MUTATION"
        ),
        "rules_golden_test",
    ),
    (
        # The bootstrap prior's softmax input.
        "native/src/prior.cpp",
        "double vacancy_potential(const PieceSet& occupied, const PieceSet& target) {",
        (
            "double vacancy_potential(const PieceSet& occupied, const PieceSet& target) {\n"
            "    if (occupied.count() > 3) return 1.0;  // MUTATION: flatten the potential"
        ),
        "rules_golden_test",
    ),
    (
        # Soo's scalar must flip once per edge; a 2P game is zero-sum.
        "native/src/mcts.cpp",
        "        value = -value;",
        (
            "        // MUTATION: do not flip the scalar across an edge\n"
            "        value = value;"
        ),
        "mcts_golden_test",
    ),
    (
        # PUCT's tie-break direction, which only bites where scores tie exactly.
        "native/src/mcts.cpp",
        "            (score == best_score && edge.action < best_action)) {",
        "            (score == best_score && edge.action > best_action)) {  // MUTATION",
        "mcts_golden_test",
    ),
    (
        # The Dirichlet mixture, invisible to any deterministic gate.
        "native/src/mcts.cpp",
        "        edge.prior = (1.0 - epsilon) * edge.prior + epsilon * (noise_[offset] / total);",
        (
            "        edge.prior = epsilon * edge.prior + (1.0 - epsilon)"
            " * (noise_[offset] / total);  // MUTATION"
        ),
        "mcts_stochastic_test",
    ),
    (
        # A wrong-but-plausible distribution: exponential instead of gamma. The
        # mean survives it; the shape does not, and only the KS check against
        # the analytic CDF can tell the difference.
        "native/include/soo/random.hpp",
        "        const double d = alpha - 1.0 / 3.0;",
        (
            "        if (alpha > 0.0) return -alpha * std::log(uniform01_open());  // MUTATION\n"
            "        const double d = alpha - 1.0 / 3.0;"
        ),
        "mcts_stochastic_test",
    ),
    (
        "native/src/topology_io.cpp",
        "    loaded.configured = true;",
        (
            "    loaded.pairwise[0][1] = 99;  // MUTATION: corrupt one distance\n"
            "    loaded.configured = true;"
        ),
        "topology_test",
    ),
    (
        # The seat mapping: the evaluator answers in the node's canonical order
        # and the tree stores by seat. Rotating that is the 3P mistake that
        # looks entirely plausible.
        "native/src/mcts3p.cpp",
        "        value[static_cast<size_t>(seat)] = pending_outcome_.value[index];",
        (
            "        value[static_cast<size_t>((seat + 1) % match_.count)] ="
            " pending_outcome_.value[index];  // MUTATION"
        ),
        "mcts3p_golden_test",
    ),
    (
        # A node maximising somebody else's component.
        "native/src/mcts3p.cpp",
        (
            "        const double score = edge.q(node.seat) + exploration_bonus(edge.prior,"
            " parent_visits,"
        ),
        (
            "        const double score = edge.q((node.seat + 1) % match_.count)"
            " + exploration_bonus(edge.prior, parent_visits,  // MUTATION"
        ),
        "mcts3p_golden_test",
    ),
    (
        # Negating the vector on the way up, as the 2P search does. A
        # three-player game is not zero-sum between two sides.
        "native/src/mcts3p.cpp",
        "            edge.value_sum[seat] += value[seat];",
        "            edge.value_sum[seat] -= value[seat];  // MUTATION",
        "mcts3p_golden_test",
    ),
    (
        # The placement utility, which only positions with a finished seat see --
        # and there are seven of those in the whole corpus.
        "native/src/mcts3p.cpp",
        "    static constexpr double kPlacement[3] = {1.0, 0.0, -1.0};",
        "    static constexpr double kPlacement[3] = {1.0, 0.0, 0.0};  // MUTATION",
        "mcts3p_golden_test",
    ),
]
