#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#pragma push_macro("MAX")
#pragma push_macro("MIN")
#undef MAX
#undef MIN

#include <lemon/bits/graph_extender.h>  // NOLINT(build/include_order)
#include <lemon/core.h>                 // NOLINT(build/include_order)

#pragma pop_macro("MIN")
#pragma pop_macro("MAX")

namespace emdgrid {

namespace detail {

/// Node and arc layout of a complete bipartite digraph with implicit arcs.
///
/// LEMON ships FullDigraph, whose arcs are implicit but which connects every
/// node to every node; a transportation problem needs the bipartite version,
/// which LEMON does not provide. This mirrors LEMON's own FullDigraphBase
/// (lemon/full_graph.h) with a source side and a target side, so the graph
/// itself costs constant memory and constant construction time no matter how
/// many arcs it denotes. POT's FullBipartiteDigraph plays the same role for
/// its bundled network simplex.
///
/// Nodes are numbered with the `n_sources` source nodes first and the
/// `n_targets` target nodes after them. Arc `(i, j)` has id
/// `i * n_targets + j`, so an arc's endpoints are recovered by a division and
/// a remainder rather than by a lookup.
///
/// This is a LEMON digraph implementation, so the member names below are the
/// ones the Digraph concept dictates rather than the project's own
/// convention. Wrap it in lemon::DigraphExtender (see FullBipartiteDigraph)
/// to obtain the iterators and maps LEMON's algorithms expect.
class FullBipartiteDigraphBase {
 public:
  using Digraph = FullBipartiteDigraphBase;

  class Node;
  class Arc;

 protected:
  int m_src_num{0};
  int m_tgt_num{0};
  int m_node_num{0};
  int m_arc_num{0};

  FullBipartiteDigraphBase() = default;

  void construct(int n_sources, int n_targets) {
    if (n_sources < 0 || n_targets < 0) {
      throw std::invalid_argument("bipartite side sizes must be nonnegative");
    }
    const int64_t arcs = static_cast<int64_t>(n_sources) * n_targets;
    constexpr auto max_arc_id = std::numeric_limits<int>::max();
    if (arcs > static_cast<int64_t>(max_arc_id)) {
      throw std::overflow_error(
          "complete bipartite graph exceeds LEMON's arc-id range");
    }
    m_src_num = n_sources;
    m_tgt_num = n_targets;
    m_node_num = n_sources + n_targets;
    m_arc_num = static_cast<int>(arcs);
  }

 public:
  using NodeNumTag = lemon::True;
  using ArcNumTag = lemon::True;
  using FindArcTag = lemon::True;

  [[nodiscard]] int nodeNum() const { return m_node_num; }
  [[nodiscard]] int arcNum() const { return m_arc_num; }

  [[nodiscard]] int maxNodeId() const { return m_node_num - 1; }
  [[nodiscard]] int maxArcId() const { return m_arc_num - 1; }

  [[nodiscard]] static int id(Node node) { return node.m_id; }
  [[nodiscard]] static int id(Arc arc) { return arc.m_id; }

  [[nodiscard]] static Node nodeFromId(int id) { return Node(id); }
  [[nodiscard]] static Arc arcFromId(int id) { return Arc(id); }

  [[nodiscard]] Node source(Arc arc) const {
    return Node(arc.m_id / m_tgt_num);
  }

  [[nodiscard]] Node target(Arc arc) const {
    return Node(m_src_num + (arc.m_id % m_tgt_num));
  }

  /// Position of an arc's tail within the source side, in `[0, n_sources)`.
  [[nodiscard]] int source_index(Arc arc) const {
    return arc.m_id / m_tgt_num;
  }

  /// Position of an arc's head within the target side, in `[0, n_targets)`.
  [[nodiscard]] int target_index(Arc arc) const {
    return arc.m_id % m_tgt_num;
  }

  /// The node holding position `ix` on the source side.
  ///
  /// Static because the source nodes are numbered first, so their ids do not
  /// depend on how the graph was sized; call it as `Graph::source_node(ix)`.
  [[nodiscard]] static Node source_node(int ix) { return Node(ix); }

  /// The node holding position `ix` on the target side.
  [[nodiscard]] Node target_node(int ix) const {
    return Node(m_src_num + ix);
  }

  [[nodiscard]] Arc arc(const Node& s, const Node& t) const {
    return Arc((s.m_id * m_tgt_num) + (t.m_id - m_src_num));
  }

  [[nodiscard]] Arc findArc(Node s, Node t, Arc prev = lemon::INVALID) const {
    if (prev != lemon::INVALID || s.m_id >= m_src_num || t.m_id < m_src_num) {
      return {lemon::INVALID};
    }
    return arc(s, t);
  }

  class Node {
    friend class FullBipartiteDigraphBase;

   protected:
    int m_id;
    explicit Node(int id) : m_id(id) {}

   public:
    Node() = default;
    Node(lemon::Invalid /*unused*/) : m_id(-1) {}  // NOLINT(*)
    bool operator==(const Node node) const { return m_id == node.m_id; }
    bool operator!=(const Node node) const { return m_id != node.m_id; }
    bool operator<(const Node node) const { return m_id < node.m_id; }
  };

  class Arc {
    friend class FullBipartiteDigraphBase;

   protected:
    int m_id;  // n_targets * source_index + target_index
    explicit Arc(int id) : m_id(id) {}

   public:
    Arc() = default;
    Arc(lemon::Invalid /*unused*/) : m_id(-1) {}  // NOLINT(*)
    bool operator==(const Arc arc) const { return m_id == arc.m_id; }
    bool operator!=(const Arc arc) const { return m_id != arc.m_id; }
    bool operator<(const Arc arc) const { return m_id < arc.m_id; }
  };

  // LEMON iterates downwards and lets the id fall to -1, which compares equal
  // to INVALID and so ends the iteration without a separate bound check.
  void first(Node& node) const { node.m_id = m_node_num - 1; }
  static void next(Node& node) { --node.m_id; }

  void first(Arc& arc) const { arc.m_id = m_arc_num - 1; }
  static void next(Arc& arc) { --arc.m_id; }

  void firstOut(Arc& arc, const Node& node) const {
    if (node.m_id >= m_src_num || m_tgt_num == 0) {
      arc.m_id = -1;
      return;
    }
    arc.m_id = ((node.m_id + 1) * m_tgt_num) - 1;
  }

  void nextOut(Arc& arc) const {
    if (arc.m_id % m_tgt_num == 0) {
      arc.m_id = 0;
    }
    --arc.m_id;
  }

  void firstIn(Arc& arc, const Node& node) const {
    if (node.m_id < m_src_num || m_src_num == 0) {
      arc.m_id = -1;
      return;
    }
    arc.m_id = m_arc_num - m_tgt_num + (node.m_id - m_src_num);
  }

  void nextIn(Arc& arc) const {
    arc.m_id -= m_tgt_num;
    if (arc.m_id < 0) {
      arc.m_id = -1;
    }
  }
};

using ExtendedFullBipartiteDigraphBase =
    lemon::DigraphExtender<FullBipartiteDigraphBase>;

/// Complete bipartite digraph usable as the graph of a LEMON algorithm.
///
/// Conforms to LEMON's Digraph concept and adds the iterators and the
/// `NodeMap` / `ArcMap` templates through lemon::DigraphExtender. Constructing
/// it is O(1): the `n_sources * n_targets` arcs are never materialised.
class FullBipartiteDigraph : public ExtendedFullBipartiteDigraphBase {
 public:
  /// @param n_sources Number of nodes on the supply side.
  /// @param n_targets Number of nodes on the demand side.
  /// @throws std::overflow_error if the arc count exceeds LEMON's int ids.
  FullBipartiteDigraph(int n_sources, int n_targets) {
    construct(n_sources, n_targets);
  }
};

/// Read-only LEMON arc map that evaluates `Fn` instead of storing values.
///
/// LEMON's min-cost-flow classes take their cost map by const reference and
/// only ever read it, so any type with the `Key` / `Value` typedefs and an
/// `operator[]` satisfies them. Wrapping a functor in this map is what lets
/// the caller skip building an arc-indexed cost container of its own.
template <class Graph, class Value_, class Fn>
class LazyArcMap {
 public:
  using Key = Graph::Arc;
  using Value = Value_;

  explicit LazyArcMap(Fn fn) : m_fn(std::move(fn)) {}

  [[nodiscard]] Value operator[](const Key& arc) const { return m_fn(arc); }

 private:
  Fn m_fn;
};

/// Deduces the functor type of a LazyArcMap from a lambda.
template <class Graph, class Value, class Fn>
[[nodiscard]] LazyArcMap<Graph, Value, std::decay_t<Fn>> make_lazy_arc_map(
    Fn&& fn) {
  return LazyArcMap<Graph, Value, std::decay_t<Fn>>(std::forward<Fn>(fn));
}

}  // namespace detail

}  // namespace emdgrid
