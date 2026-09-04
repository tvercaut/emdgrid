/* -*- mode: C++; indent-tabs-mode: nil; -*-
 *
 * Sparse bipartite graph for optimal transport
 * Only stores edges that are explicitly added (not all n1×n2 edges)
 *
 * Uses CSR (Compressed Sparse Row) format for better cache locality and performance
 * - Binary search for arc lookup: O(log k) where k = avg edges per node
 * - Compact memory layout for better cache performance
 * - Requires edges to be provided in sorted order during construction
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace potlemon {

struct Invalid {
  bool operator==(Invalid /*unused*/) const { return true; }
  bool operator!=(Invalid /*unused*/) const { return false; }
  bool operator<(Invalid /*unused*/) const { return false; }
};

inline constexpr Invalid INVALID_OBJ = Invalid();

class SparseBipartiteDigraphBase {
 public:
  typedef SparseBipartiteDigraphBase Digraph;
  typedef int Node;
  typedef int64_t Arc;

 protected:
  int m_node_num;
  int64_t m_arc_num;
  int m_n1;
  int m_n2;

  std::vector<Node> m_arc_sources;
  std::vector<Node> m_arc_targets;

  // CSR format
  std::vector<int64_t> m_row_ptr;
  std::vector<Node> m_col_indices;
  std::vector<Arc> m_arc_ids;

  mutable std::vector<std::vector<Arc>> m_in_arcs;
  mutable bool m_in_arcs_built;

  // Position tracking for O(1) iteration
  mutable std::vector<int64_t> m_arc_to_out_pos;
  mutable std::vector<int64_t> m_arc_to_in_pos;
  mutable bool m_position_maps_built;

  SparseBipartiteDigraphBase()
      : m_node_num(0),
        m_arc_num(0),
        m_n1(0),
        m_n2(0),
        m_in_arcs_built(false),
        m_position_maps_built(false) {}

  void construct(int n1, int n2) {
    m_node_num = n1 + n2;
    m_n1 = n1;
    m_n2 = n2;
    m_arc_num = 0;
    m_arc_sources.clear();
    m_arc_targets.clear();
    m_row_ptr.clear();
    m_col_indices.clear();
    m_arc_ids.clear();
    m_in_arcs.clear();
    m_in_arcs_built = false;
    m_arc_to_out_pos.clear();
    m_arc_to_in_pos.clear();
    m_position_maps_built = false;
  }

  void build_in_arcs() const {
    if (m_in_arcs_built) {
      return;
    }

    m_in_arcs.resize(m_node_num);

    for (Arc a = 0; a < m_arc_num; ++a) {
      Node tgt = m_arc_targets[a];
      m_in_arcs[tgt].push_back(a);
    }

    m_in_arcs_built = true;
  }

  void build_position_maps() const {
    if (m_position_maps_built) {
      return;
    }

    m_arc_to_out_pos.resize(m_arc_num);
    m_arc_to_in_pos.resize(m_arc_num);

    for (int64_t pos = 0; pos < m_arc_num; ++pos) {
      Arc arc_id = m_arc_ids[pos];
      m_arc_to_out_pos[arc_id] = pos;
    }

    build_in_arcs();
    for (Node node = 0; node < m_node_num; ++node) {
      const std::vector<Arc>& in = m_in_arcs[node];
      for (size_t pos = 0; pos < in.size(); ++pos) {
        Arc arc_id = in[pos];
        m_arc_to_in_pos[arc_id] = static_cast<int64_t>(pos);
      }
    }

    m_position_maps_built = true;
  }

 public:
  virtual Node operator()(int ix) const { return Node(ix); }
  static int index(const Node& node) { return node; }

  virtual void buildFromEdges(const std::vector<std::pair<Node, Node>>& edges) {
    m_arc_num = static_cast<int64_t>(edges.size());

    if (m_arc_num == 0) {
      m_row_ptr.assign(m_n1 + 1, 0);
      return;
    }

    std::vector<std::tuple<Node, Node, Arc>> indexed_edges;
    indexed_edges.reserve(m_arc_num);
    for (Arc i = 0; i < m_arc_num; ++i) {
      indexed_edges.emplace_back(edges[i].first, edges[i].second, i);
    }

    std::sort(indexed_edges.begin(), indexed_edges.end(),
              [](const auto& a, const auto& b) {
                if (std::get<0>(a) != std::get<0>(b)) {
                  return std::get<0>(a) < std::get<0>(b);
                }
                return std::get<1>(a) < std::get<1>(b);
              });

    m_arc_sources.resize(m_arc_num);
    m_arc_targets.resize(m_arc_num);
    m_col_indices.resize(m_arc_num);
    m_arc_ids.resize(m_arc_num);
    m_row_ptr.resize(m_n1 + 1);

    m_row_ptr[0] = 0;
    int current_row = 0;

    for (int64_t i = 0; i < m_arc_num; ++i) {
      Node src = std::get<0>(indexed_edges[i]);
      Node tgt = std::get<1>(indexed_edges[i]);
      Arc orig_arc_id = std::get<2>(indexed_edges[i]);

      while (current_row < src) {
        m_row_ptr[++current_row] = i;
      }

      m_arc_sources[orig_arc_id] = src;
      m_arc_targets[orig_arc_id] = tgt;
      m_col_indices[i] = tgt;
      m_arc_ids[i] = orig_arc_id;
    }

    while (current_row < m_n1) {
      m_row_ptr[++current_row] = m_arc_num;
    }

    m_in_arcs_built = false;
  }

  virtual Arc arc(const Node& s, const Node& t) const {
    if (s < 0 || s >= m_n1 || t < m_n1 || t >= m_node_num) {
      return Arc(-1);
    }

    int64_t start = m_row_ptr[s];
    int64_t end = m_row_ptr[s + 1];

    auto it = std::lower_bound(m_col_indices.begin() + start,
                               m_col_indices.begin() + end, t);

    if (it != m_col_indices.begin() + end && *it == t) {
      int64_t pos = it - m_col_indices.begin();
      return m_arc_ids[pos];
    }

    return Arc(-1);
  }

  virtual int nodeNum() const { return m_node_num; }
  virtual int64_t arcNum() const { return m_arc_num; }

  int maxNodeId() const { return m_node_num - 1; }
  int64_t maxArcId() const { return m_arc_num - 1; }

  Node source(Arc arc) const {
    return (arc >= 0 && arc < m_arc_num) ? m_arc_sources[arc] : Node(-1);
  }

  Node target(Arc arc) const {
    return (arc >= 0 && arc < m_arc_num) ? m_arc_targets[arc] : Node(-1);
  }

  static int id(Node node) { return node; }
  static int64_t id(Arc arc) { return arc; }

  static Node nodeFromId(int id) { return Node(id); }
  static Arc arcFromId(int64_t id) { return Arc(id); }

  Arc findArc(Node s, Node t, Arc prev = -1) const {
    return prev == -1 ? arc(s, t) : Arc(-1);
  }

  void first(Node& node) const { node = m_node_num - 1; }

  static void next(Node& node) { --node; }

  void first(Arc& arc) const { arc = m_arc_num - 1; }

  static void next(Arc& arc) { --arc; }

  void firstOut(Arc& arc, const Node& node) const {
    if (node < 0 || node >= m_n1) {
      arc = -1;
      return;
    }

    int64_t start = m_row_ptr[node];
    int64_t end = m_row_ptr[node + 1];

    arc = (start < end) ? m_arc_ids[start] : Arc(-1);
  }

  void nextOut(Arc& arc) const {
    if (arc < 0) {
      return;
    }

    build_position_maps();

    int64_t pos = m_arc_to_out_pos[arc];
    Node src = m_arc_sources[arc];
    int64_t end = m_row_ptr[src + 1];

    arc = (pos + 1 < end) ? m_arc_ids[pos + 1] : Arc(-1);
  }

  void firstIn(Arc& arc, const Node& node) const {
    build_in_arcs();

    if (node < 0 || node >= m_node_num || node < m_n1) {
      arc = -1;
      return;
    }

    const std::vector<Arc>& in = m_in_arcs[node];
    arc = in.empty() ? Arc(-1) : in[0];
  }

  void nextIn(Arc& arc) const {
    if (arc < 0) {
      return;
    }

    build_position_maps();

    int64_t pos = m_arc_to_in_pos[arc];
    Node tgt = m_arc_targets[arc];
    const std::vector<Arc>& in = m_in_arcs[tgt];

    arc = (pos + 1 < static_cast<int64_t>(in.size()))
              ? in[static_cast<std::size_t>(pos + 1)]
              : Arc(-1);
  }
};

class SparseBipartiteDigraph : public SparseBipartiteDigraphBase {
  typedef SparseBipartiteDigraphBase Parent;

 public:
  SparseBipartiteDigraph() { construct(0, 0); }

  SparseBipartiteDigraph(int n1, int n2) { construct(n1, n2); }

  Node operator()(int ix) const { return Parent::operator()(ix); }
  static int index(const Node& node) { return Parent::index(node); }

  void buildFromEdges(const std::vector<std::pair<Node, Node>>& edges) {
    Parent::buildFromEdges(edges);
  }

  Arc arc(const Node& s, const Node& t) const { return Parent::arc(s, t); }

  int nodeNum() const { return Parent::nodeNum(); }
  int64_t arcNum() const { return Parent::arcNum(); }
};

}  // namespace potlemon
