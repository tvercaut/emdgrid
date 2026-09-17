/* -*- mode: C++; indent-tabs-mode: nil; -*-
 *
 * Complete bipartite digraph with implicit arcs, for the potlemon network
 * simplex. No arc is ever materialised: the n1 x n2 arcs are denoted by their
 * ids alone, so the graph costs constant memory and constant construction
 * time.
 *
 * Adapted from full_bipartitegraph.h in POT
 * (https://github.com/PythonOT/POT/blob/master/ot/lp/full_bipartitegraph.h),
 * itself adapted by Nicolas Bonneel (2013) from full_graph.h in LEMON.
 *
 **** Original file Copyright Notice :
 * Copyright (C) 2003-2010
 * Egervary Jeno Kombinatorikus Optimalizalasi Kutatocsoport
 * (Egervary Research Group on Combinatorial Optimization, EGRES).
 *
 * Permission to use, modify and distribute this software is granted
 * provided that this copyright notice appears in all copies. For
 * precise terms see the accompanying LICENSE file.
 *
 * This software is provided "AS IS" with no warranty of any kind,
 * express or implied, and with no claim as to its suitability for any
 * purpose.
 *
 * Adaptations for emdgrid:
 *   - namespace potlemon, C++20, project naming conventions
 *   - interface aligned with SparseBipartiteDigraph so that either graph can
 *     drive NetworkSimplexSimple
 */

#pragma once

#include <cstdint>
#include <stdexcept>

namespace potlemon {

/// Node and arc layout of a complete bipartite digraph with implicit arcs.
///
/// Nodes are numbered with the `n1` source nodes first and the `n2` target
/// nodes after them. Arc `(i, j)` has id `i * n2 + j`, so an arc's endpoints
/// are a division and a remainder rather than a lookup, and neither endpoint
/// nor adjacency arrays are stored.
///
/// NetworkSimplexSimple relies on exactly this id layout when its
/// `EndpointStorageMode::ArcEndpoints` is in force: it reconstructs endpoints
/// with the same arithmetic instead of reading its `_source` / `_target`
/// arrays. Changing the layout here without changing `arcSource` / `arcTarget`
/// there would silently solve a different problem.
class FullBipartiteDigraphBase {
 public:
  typedef FullBipartiteDigraphBase Digraph;
  typedef int Node;
  typedef int64_t Arc;

 protected:
  int m_node_num{0};
  int64_t m_arc_num{0};
  int m_n1{0};
  int m_n2{0};

  FullBipartiteDigraphBase() = default;

  void construct(int n1, int n2) {
    if (n1 < 0 || n2 < 0) {
      throw std::invalid_argument("bipartite side sizes must be nonnegative");
    }
    m_n1 = n1;
    m_n2 = n2;
    m_node_num = n1 + n2;
    m_arc_num = static_cast<int64_t>(n1) * static_cast<int64_t>(n2);
  }

 public:
  int nodeNum() const { return m_node_num; }
  int64_t arcNum() const { return m_arc_num; }

  int maxNodeId() const { return m_node_num - 1; }
  int64_t maxArcId() const { return m_arc_num - 1; }

  /// Number of nodes on the supply side.
  int n1() const { return m_n1; }
  /// Number of nodes on the demand side.
  int n2() const { return m_n2; }

  static int id(Node node) { return node; }
  static int64_t id(Arc arc) { return arc; }
  static Node nodeFromId(int i) { return Node(i); }
  static Arc arcFromId(int64_t i) { return Arc(i); }

  Node source(Arc arc) const { return static_cast<Node>(arc / m_n2); }
  Node target(Arc arc) const {
    return static_cast<Node>(arc % m_n2) + m_n1;
  }

  Arc arc(const Node& s, const Node& t) const {
    if (s >= m_n1 || t < m_n1) {
      return Arc(-1);
    }
    return Arc((static_cast<int64_t>(s) * m_n2) + (t - m_n1));
  }

  Arc findArc(Node s, Node t, Arc prev = -1) const {
    return prev == -1 ? arc(s, t) : Arc(-1);
  }

  // Iteration runs downwards and lets the id fall to -1, which is INVALID_ARC
  // / INVALID_NODE, so no separate bound check is needed. This matches
  // SparseBipartiteDigraph.
  void first(Node& node) const { node = m_node_num - 1; }
  static void next(Node& node) { --node; }

  void first(Arc& arc) const { arc = m_arc_num - 1; }
  static void next(Arc& arc) { --arc; }

  void firstOut(Arc& arc, const Node& node) const {
    if (node < 0 || node >= m_n1 || m_n2 == 0) {
      arc = -1;
      return;
    }
    arc = (static_cast<int64_t>(node + 1) * m_n2) - 1;
  }

  void nextOut(Arc& arc) const {
    if (arc % m_n2 == 0) {
      arc = 0;
    }
    --arc;
  }

  void firstIn(Arc& arc, const Node& node) const {
    if (node < m_n1 || node >= m_node_num || m_n1 == 0) {
      arc = -1;
      return;
    }
    arc = m_arc_num - m_n2 + (node - m_n1);
  }

  void nextIn(Arc& arc) const {
    arc -= m_n2;
    if (arc < 0) {
      arc = -1;
    }
  }

  /// No-op: there are no auxiliary structures to build.
  ///
  /// SparseBipartiteDigraph lazily builds in-arc lists and position maps, and
  /// NetworkSimplexSimple calls this before its OpenMP loops so that the
  /// building happens serially. Nothing is lazily built here, but the entry
  /// point has to exist for either graph to drive the same solver.
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  void ensureAuxStructuresBuilt() const {}
};

/// Complete bipartite digraph over `n1` sources and `n2` targets.
///
/// Construction is O(1) and the object holds four integers: the arcs exist
/// only as ids. Pair it with NetworkSimplexSimple's lazy storage modes to
/// solve a dense transportation problem in memory proportional to the number
/// of nodes rather than the number of arcs.
class FullBipartiteDigraph : public FullBipartiteDigraphBase {
 public:
  FullBipartiteDigraph() { construct(0, 0); }

  /// @param n1 Number of nodes on the supply side.
  /// @param n2 Number of nodes on the demand side.
  FullBipartiteDigraph(int n1, int n2) { construct(n1, n2); }
};

}  // namespace potlemon
