/* -*- mode: C++; indent-tabs-mode: nil; -*-
 * Sparse bipartite graph for optimal transport
 * Adapted for emdgrid under potlemon namespace.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace potlemon {

struct Invalid {
  bool operator==(Invalid) const { return true; }
  bool operator!=(Invalid) const { return false; }
  bool operator<(Invalid) const { return false; }
};

inline constexpr Invalid INVALID_OBJ = Invalid();

#ifndef POTLEMON_DIGRAPH_TYPEDEFS
#define POTLEMON_DIGRAPH_TYPEDEFS(Digraph) \
  typedef Digraph::Node Node;              \
  typedef Digraph::Arc Arc;
#endif

#ifndef POTLEMON_TEMPLATE_DIGRAPH_TYPEDEFS
#define POTLEMON_TEMPLATE_DIGRAPH_TYPEDEFS(Digraph) \
  typedef typename Digraph::Node Node;              \
  typedef typename Digraph::Arc Arc;
#endif

class SparseBipartiteDigraphBase {
 public:
  typedef SparseBipartiteDigraphBase Digraph;
  typedef int Node;
  typedef int64_t Arc;

 protected:
  int _node_num;
  int64_t _arc_num;
  int _n1, _n2;

  std::vector<Node> _arc_sources;  // _arc_sources[arc_id] = source node
  std::vector<Node> _arc_targets;  // _arc_targets[arc_id] = target node

  // CSR format
  std::vector<int64_t> _row_ptr;
  std::vector<Node> _col_indices;
  std::vector<Arc> _arc_ids;

  mutable std::vector<std::vector<Arc>> _in_arcs;
  mutable bool _in_arcs_built;

  // Position tracking for O(1) iteration
  mutable std::vector<int64_t> _arc_to_out_pos;
  mutable std::vector<int64_t> _arc_to_in_pos;
  mutable bool _position_maps_built;

  SparseBipartiteDigraphBase()
      : _node_num(0),
        _arc_num(0),
        _n1(0),
        _n2(0),
        _in_arcs_built(false),
        _position_maps_built(false) {}

  void construct(int n1, int n2) {
    _node_num = n1 + n2;
    _n1 = n1;
    _n2 = n2;
    _arc_num = 0;
    _arc_sources.clear();
    _arc_targets.clear();
    _row_ptr.clear();
    _col_indices.clear();
    _arc_ids.clear();
    _in_arcs.clear();
    _in_arcs_built = false;
    _arc_to_out_pos.clear();
    _arc_to_in_pos.clear();
    _position_maps_built = false;
  }

  void build_in_arcs() const {
    if (_in_arcs_built) return;

    _in_arcs.resize(_node_num);

    for (Arc a = 0; a < _arc_num; ++a) {
      Node tgt = _arc_targets[a];
      _in_arcs[tgt].push_back(a);
    }

    _in_arcs_built = true;
  }

  void build_position_maps() const {
    if (_position_maps_built) return;

    _arc_to_out_pos.resize(_arc_num);
    _arc_to_in_pos.resize(_arc_num);

    for (int64_t pos = 0; pos < _arc_num; ++pos) {
      Arc arc_id = _arc_ids[pos];
      _arc_to_out_pos[arc_id] = pos;
    }

    build_in_arcs();
    for (Node node = 0; node < _node_num; ++node) {
      const std::vector<Arc>& in = _in_arcs[node];
      for (size_t pos = 0; pos < in.size(); ++pos) {
        Arc arc_id = in[pos];
        _arc_to_in_pos[arc_id] = pos;
      }
    }

    _position_maps_built = true;
  }

 public:
  Node operator()(int ix) const { return Node(ix); }
  static int index(const Node& node) { return node; }

  void buildFromEdges(const std::vector<std::pair<Node, Node>>& edges) {
    _arc_num = edges.size();

    if (_arc_num == 0) {
      _row_ptr.assign(_n1 + 1, 0);
      return;
    }

    std::vector<std::tuple<Node, Node, Arc>> indexed_edges;
    indexed_edges.reserve(_arc_num);
    for (Arc i = 0; i < _arc_num; ++i) {
      indexed_edges.emplace_back(edges[i].first, edges[i].second, i);
    }

    std::sort(indexed_edges.begin(), indexed_edges.end(),
              [](const auto& a, const auto& b) {
                if (std::get<0>(a) != std::get<0>(b))
                  return std::get<0>(a) < std::get<0>(b);
                return std::get<1>(a) < std::get<1>(b);
              });

    _arc_sources.resize(_arc_num);
    _arc_targets.resize(_arc_num);
    _col_indices.resize(_arc_num);
    _arc_ids.resize(_arc_num);
    _row_ptr.resize(_n1 + 1);

    _row_ptr[0] = 0;
    int current_row = 0;

    for (int64_t i = 0; i < _arc_num; ++i) {
      Node src = std::get<0>(indexed_edges[i]);
      Node tgt = std::get<1>(indexed_edges[i]);
      Arc orig_arc_id = std::get<2>(indexed_edges[i]);

      while (current_row < src) {
        _row_ptr[++current_row] = i;
      }

      _arc_sources[orig_arc_id] = src;
      _arc_targets[orig_arc_id] = tgt;
      _col_indices[i] = tgt;
      _arc_ids[i] = orig_arc_id;
    }

    while (current_row < _n1) {
      _row_ptr[++current_row] = _arc_num;
    }

    _in_arcs_built = false;
  }

  Arc arc(const Node& s, const Node& t) const {
    if (s < 0 || s >= _n1 || t < _n1 || t >= _node_num) {
      return Arc(-1);
    }

    int64_t start = _row_ptr[s];
    int64_t end = _row_ptr[s + 1];

    auto it = std::lower_bound(_col_indices.begin() + start,
                               _col_indices.begin() + end, t);

    if (it != _col_indices.begin() + end && *it == t) {
      int64_t pos = it - _col_indices.begin();
      return _arc_ids[pos];
    }

    return Arc(-1);
  }

  int nodeNum() const { return _node_num; }
  int64_t arcNum() const { return _arc_num; }

  int maxNodeId() const { return _node_num - 1; }
  int64_t maxArcId() const { return _arc_num - 1; }

  Node source(Arc arc) const {
    return (arc >= 0 && arc < _arc_num) ? _arc_sources[arc] : Node(-1);
  }

  Node target(Arc arc) const {
    return (arc >= 0 && arc < _arc_num) ? _arc_targets[arc] : Node(-1);
  }

  static int id(Node node) { return node; }
  static int64_t id(Arc arc) { return arc; }

  static Node nodeFromId(int id) { return Node(id); }
  static Arc arcFromId(int64_t id) { return Arc(id); }

  Arc findArc(Node s, Node t, Arc prev = -1) const {
    return prev == -1 ? arc(s, t) : Arc(-1);
  }

  void first(Node& node) const { node = _node_num - 1; }

  static void next(Node& node) { --node; }

  void first(Arc& arc) const { arc = _arc_num - 1; }

  static void next(Arc& arc) { --arc; }

  void firstOut(Arc& arc, const Node& node) const {
    if (node < 0 || node >= _n1) {
      arc = -1;
      return;
    }

    int64_t start = _row_ptr[node];
    int64_t end = _row_ptr[node + 1];

    arc = (start < end) ? _arc_ids[start] : Arc(-1);
  }

  void nextOut(Arc& arc) const {
    if (arc < 0) return;

    build_position_maps();

    int64_t pos = _arc_to_out_pos[arc];
    Node src = _arc_sources[arc];
    int64_t end = _row_ptr[src + 1];

    arc = (pos + 1 < end) ? _arc_ids[pos + 1] : Arc(-1);
  }

  void firstIn(Arc& arc, const Node& node) const {
    build_in_arcs();

    if (node < 0 || node >= _node_num || node < _n1) {
      arc = -1;
      return;
    }

    const std::vector<Arc>& in = _in_arcs[node];
    arc = in.empty() ? Arc(-1) : in[0];
  }

  void nextIn(Arc& arc) const {
    if (arc < 0) return;

    build_position_maps();

    int64_t pos = _arc_to_in_pos[arc];
    Node tgt = _arc_targets[arc];
    const std::vector<Arc>& in = _in_arcs[tgt];

    arc = (pos + 1 < in.size()) ? in[pos + 1] : Arc(-1);
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

  Arc arc(Node s, Node t) const { return Parent::arc(s, t); }

  int nodeNum() const { return Parent::nodeNum(); }
  int64_t arcNum() const { return Parent::arcNum(); }
};

}  // namespace potlemon
