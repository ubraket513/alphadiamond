// Flat search-tree arena.
//
// Deliberately not a transliteration of ScalarNode/ScalarEdge: Python's dicts
// become index ranges into two vectors, and `total_visits` is a cached
// aggregate instead of re-summing the children on every selection.  Both are
// representation changes only -- the visit sum is over integers, so caching it
// is exact, and Gate B pins the observable behaviour either way.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "soo/state.hpp"

namespace soo {

inline constexpr int32_t kNoChild = -1;

struct Edge {
    int32_t action = 0;    // canonical action id
    int32_t child = kNoChild;
    double prior = 0.0;
    double value_sum = 0.0;
    uint32_t visits = 0;

    // ScalarEdge.q
    double q() const { return visits ? value_sum / static_cast<double>(visits) : 0.0; }
};

struct Node {
    State state;
    uint32_t edge_begin = 0;
    uint16_t edge_count = 0;
    uint8_t player_id = 0;   // search-adapter perspective, not state.current_player
    bool expanded = false;
    bool terminal = false;
    uint32_t total_visits = 0;  // cached sum of this node's edge visit counts
};

// One search's storage.  Cleared between moves; Python builds a fresh MCTS2P
// per move, so there is no tree reuse to preserve.
class Arena {
  public:
    void reset(size_t simulations, size_t mean_legal) {
        nodes_.clear();
        edges_.clear();
        nodes_.reserve(simulations + 2);
        edges_.reserve((simulations + 2) * mean_legal);
    }

    uint32_t add_node(const State& state, uint8_t player_id, bool terminal) {
        Node node;
        node.state = state;
        node.player_id = player_id;
        node.terminal = terminal;
        node.edge_begin = static_cast<uint32_t>(edges_.size());
        nodes_.push_back(node);
        return static_cast<uint32_t>(nodes_.size() - 1);
    }

    Node& node(uint32_t index) { return nodes_[index]; }
    const Node& node(uint32_t index) const { return nodes_[index]; }
    Edge& edge(uint32_t index) { return edges_[index]; }
    const Edge& edge(uint32_t index) const { return edges_[index]; }

    // Open this node's edge block at the end of the arena.
    //
    // Edge blocks are contiguous and append-only, which holds because a node is
    // always expanded in the same simulation that creates it (the descent
    // breaks the moment it reaches an unexpanded node).  If a future change
    // ever defers an expansion, this would silently interleave two nodes' edges
    // -- so it is checked rather than assumed.
    void open_edges(uint32_t node_index) {
        if (nodes_[node_index].edge_count != 0) {
            throw std::logic_error("node already has an edge block");
        }
        nodes_[node_index].edge_begin = static_cast<uint32_t>(edges_.size());
    }
    void push_edge(uint32_t node_index, int32_t action, double prior) {
        edges_.push_back(Edge{action, kNoChild, prior, 0.0, 0});
        ++nodes_[node_index].edge_count;
    }

    size_t node_count() const { return nodes_.size(); }

  private:
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
};

}  // namespace soo
