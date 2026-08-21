#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "emdgrid/emdgrid.hpp"

namespace emdgrid {

namespace detail {

// ---------------------------------------------------------------------------
//  Data structures for the network-simplex EMD-L1 solver
// ---------------------------------------------------------------------------

struct EmdNode {
  double d{0.0};                    // supply/demand = H1[i] - H2[i]
  int u{0};                         // dual variable (potential)
  int level{-1};                    // depth in BV spanning tree (-1=unvisited)
  std::ptrdiff_t parent_node{-1};   // index of parent node  (-1 = root)
  std::ptrdiff_t parent_edge{-1};   // index of parent BV edge (-1 = root)
  std::ptrdiff_t first_child{-1};   // head of linked-list of child BV edges
};

struct EmdEdge {
  std::ptrdiff_t p{-1};    // parent endpoint in current BV tree orientation
  std::ptrdiff_t c{-1};    // child  endpoint
  double flow{0.0};
  int dir{1};              // 1 = outward (p→c carries `flow` units), 0 = inward
  std::ptrdiff_t next{-1};  // next sibling in parent's child-edge linked list
};

// ---------------------------------------------------------------------------
//  Network-simplex solver
// ---------------------------------------------------------------------------
class EmdSolver {
 public:
  static constexpr int kDefaultMaxIter = 500;

  EmdSolver(std::size_t num_nodes, std::size_t num_edges)
      : nodes_(num_nodes),
        edges_(num_edges),
        is_bv_(num_edges, false),
        aux_(num_nodes + 4),
        from_loop_(num_nodes + 4),
        to_loop_(num_nodes + 4) {
    nbv_.reserve(num_edges);
  }

  EmdNode& node(std::ptrdiff_t i) { return nodes_[i]; }

  // Register a BV edge chosen by the greedy step.
  // `from_node` is the processed node, `to_node` is its greedy neighbour.
  void register_bv(std::ptrdiff_t from_node, std::ptrdiff_t to_node,
                   double flow, int dir) {
    std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(n_edges_++);
    edges_[idx] = {from_node, to_node, flow, dir, -1};
    is_bv_[idx] = true;
  }

  // Register a non-basic (NBV) edge from the greedy step.
  void register_nbv(std::ptrdiff_t low, std::ptrdiff_t high) {
    std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(n_edges_++);
    edges_[idx] = {low, high, 0.0, 1, -1};
    is_bv_[idx] = false;
    nbv_.push_back(idx);
  }

  // Run network simplex and return the EMD-L1 value.
  double solve(std::ptrdiff_t root, int max_iter = kDefaultMaxIter);

 private:
  void init_bv_tree(std::ptrdiff_t root);
  void update_subtree(std::ptrdiff_t start);
  bool is_optimal();
  void find_loop();
  void pivot();
  double total_flow() const;

  std::vector<EmdNode> nodes_;
  std::vector<EmdEdge> edges_;
  std::vector<bool> is_bv_;
  std::vector<std::ptrdiff_t> nbv_;
  std::size_t n_edges_{0};

  // Pivot state
  std::ptrdiff_t enter_nbv_pos_{-1};  // index into nbv_[] of entering edge
  std::ptrdiff_t leave_edge_{-1};     // edge index of leaving BV edge
  int leave_flag_{0};                 // 0 = leave in from_loop, 1 = in to_loop

  std::vector<std::ptrdiff_t> aux_;
  std::vector<std::ptrdiff_t> from_loop_;
  std::vector<std::ptrdiff_t> to_loop_;
  std::ptrdiff_t i_from_{0};
  std::ptrdiff_t i_to_{0};
};

// ---------------------------------------------------------------------------
//  Implementation
// ---------------------------------------------------------------------------

inline void EmdSolver::init_bv_tree(std::ptrdiff_t root) {
  const auto n = static_cast<std::ptrdiff_t>(nodes_.size());

  // Build BV adjacency: for each BV edge e, record it at both endpoints.
  std::vector<std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>>> adj(n);
  for (std::ptrdiff_t e = 0; e < static_cast<std::ptrdiff_t>(n_edges_); ++e) {
    if (is_bv_[e]) {
      adj[edges_[e].p].emplace_back(e, edges_[e].c);
      adj[edges_[e].c].emplace_back(e, edges_[e].p);
    }
  }

  nodes_[root].level = 0;
  nodes_[root].parent_node = -1;
  nodes_[root].parent_edge = -1;
  nodes_[root].first_child = -1;

  std::ptrdiff_t head = 0;
  std::ptrdiff_t tail = 0;
  aux_[tail++] = root;

  while (head < tail) {
    std::ptrdiff_t u = aux_[head++];
    for (auto [eidx, v] : adj[u]) {
      if (nodes_[v].level == -1) {
        // v is not yet in the tree: make v a child of u via edge eidx
        nodes_[v].level = nodes_[u].level + 1;
        nodes_[v].parent_node = u;
        nodes_[v].parent_edge = eidx;
        nodes_[v].first_child = -1;

        // Orient edge so that p == u, c == v
        if (edges_[eidx].c == u) {
          std::swap(edges_[eidx].p, edges_[eidx].c);
          edges_[eidx].dir = 1 - edges_[eidx].dir;
        }

        // Prepend eidx to u's child list
        edges_[eidx].next = nodes_[u].first_child;
        nodes_[u].first_child = eidx;

        aux_[tail++] = v;
      }
    }
  }
}

inline void EmdSolver::update_subtree(std::ptrdiff_t start) {
  std::ptrdiff_t head = 0;
  std::ptrdiff_t tail = 0;
  aux_[tail++] = start;

  while (head < tail) {
    std::ptrdiff_t u = aux_[head++];
    std::ptrdiff_t eidx = nodes_[u].first_child;
    while (eidx >= 0) {
      std::ptrdiff_t v = edges_[eidx].c;  // child node
      nodes_[v].level = nodes_[u].level + 1;
      // Outward edge (u→v): u's potential is one step above v.
      // Dual condition: c_{pq} = u[p] - u[q] = 1, so u[q] = u[p] - 1.
      nodes_[v].u =
          (edges_[eidx].dir == 1) ? (nodes_[u].u - 1) : (nodes_[u].u + 1);
      aux_[tail++] = v;
      eidx = edges_[eidx].next;
    }
  }
}

inline bool EmdSolver::is_optimal() {
  enter_nbv_pos_ = -1;
  int min_cost = 0;

  for (std::size_t k = 0; k < nbv_.size(); ++k) {
    std::ptrdiff_t eidx = nbv_[k];
    const int cp = nodes_[edges_[eidx].p].u;
    const int cc = nodes_[edges_[eidx].c].u;
    // Reduced cost p→c: 1 - u[p] + u[c]
    const int cost = 1 - cp + cc;
    if (cost < min_cost) {
      min_cost = cost;
      enter_nbv_pos_ = static_cast<std::ptrdiff_t>(k);
    }
    // Reduced cost c→p: 1 + u[p] - u[c]
    const int cost_rev = 1 + cp - cc;
    if (cost_rev < min_cost) {
      min_cost = cost_rev;
      enter_nbv_pos_ = static_cast<std::ptrdiff_t>(k);
    }
  }

  if (enter_nbv_pos_ < 0) {
    return true;  // already optimal
  }

  // Orient the entering edge so that flow goes p→c (dir = 1).
  std::ptrdiff_t eidx = nbv_[enter_nbv_pos_];
  const int cp = nodes_[edges_[eidx].p].u;
  const int cc = nodes_[edges_[eidx].c].u;
  // If the reversed direction gave the minimum cost, swap endpoints.
  if (min_cost == (1 + cp - cc)) {
    std::swap(edges_[eidx].p, edges_[eidx].c);
  }
  edges_[eidx].dir = 1;
  return false;
}

inline void EmdSolver::find_loop() {
  // Entering edge: edges_[nbv_[enter_nbv_pos_]]
  // pFrom: ancestors of entering parent; pTo: ancestors of entering child.
  std::ptrdiff_t enter_eidx = nbv_[enter_nbv_pos_];
  std::ptrdiff_t from_node = edges_[enter_eidx].p;  // entering parent
  std::ptrdiff_t to_node = edges_[enter_eidx].c;    // entering child

  i_from_ = 0;
  i_to_ = 0;
  leave_edge_ = -1;
  leave_flag_ = 0;
  double min_flow = std::numeric_limits<double>::max();

  // Bring from_node and to_node to the same tree level
  while (nodes_[from_node].level > nodes_[to_node].level) {
    std::ptrdiff_t eidx = nodes_[from_node].parent_edge;
    from_loop_[i_from_++] = eidx;
    // Inward edge on from_loop is a leaving candidate
    if (edges_[eidx].dir == 0 && edges_[eidx].flow < min_flow) {
      min_flow = edges_[eidx].flow;
      leave_edge_ = eidx;
      leave_flag_ = 0;
    }
    from_node = nodes_[from_node].parent_node;
  }

  while (nodes_[to_node].level > nodes_[from_node].level) {
    std::ptrdiff_t eidx = nodes_[to_node].parent_edge;
    to_loop_[i_to_++] = eidx;
    // Outward edge on to_loop is a leaving candidate
    if (edges_[eidx].dir == 1 && edges_[eidx].flow < min_flow) {
      min_flow = edges_[eidx].flow;
      leave_edge_ = eidx;
      leave_flag_ = 1;
    }
    to_node = nodes_[to_node].parent_node;
  }

  // Advance both until LCA is found
  while (from_node != to_node) {
    {
      std::ptrdiff_t eidx = nodes_[from_node].parent_edge;
      from_loop_[i_from_++] = eidx;
      if (edges_[eidx].dir == 0 && edges_[eidx].flow < min_flow) {
        min_flow = edges_[eidx].flow;
        leave_edge_ = eidx;
        leave_flag_ = 0;
      }
      from_node = nodes_[from_node].parent_node;
    }
    {
      std::ptrdiff_t eidx = nodes_[to_node].parent_edge;
      to_loop_[i_to_++] = eidx;
      if (edges_[eidx].dir == 1 && edges_[eidx].flow < min_flow) {
        min_flow = edges_[eidx].flow;
        leave_edge_ = eidx;
        leave_flag_ = 1;
      }
      to_node = nodes_[to_node].parent_node;
    }
  }

  // If the leaving edge is on the from_loop side, reverse the entering edge.
  if (leave_flag_ == 0) {
    std::ptrdiff_t eidx = nbv_[enter_nbv_pos_];
    std::swap(edges_[eidx].p, edges_[eidx].c);
    edges_[eidx].dir = 1 - edges_[eidx].dir;
  }
}

inline void EmdSolver::pivot() {
  std::ptrdiff_t enter_eidx = nbv_[enter_nbv_pos_];
  const double min_flow = edges_[leave_edge_].flow;

  // Update flows along the loop
  for (std::ptrdiff_t k = 0; k < i_from_; ++k) {
    std::ptrdiff_t eidx = from_loop_[k];
    if (edges_[eidx].dir == 1) {
      edges_[eidx].flow += min_flow;
    } else {
      edges_[eidx].flow -= min_flow;
    }
  }
  for (std::ptrdiff_t k = 0; k < i_to_; ++k) {
    std::ptrdiff_t eidx = to_loop_[k];
    if (edges_[eidx].dir == 1) {
      edges_[eidx].flow -= min_flow;
    } else {
      edges_[eidx].flow += min_flow;
    }
  }

  // Remove leaving BV edge from its parent's child list
  std::ptrdiff_t leave_parent = edges_[leave_edge_].p;
  std::ptrdiff_t leave_child = edges_[leave_edge_].c;

  std::ptrdiff_t* child_ptr = &nodes_[leave_parent].first_child;
  while (*child_ptr != leave_edge_) {
    child_ptr = &edges_[*child_ptr].next;
  }
  *child_ptr = edges_[leave_edge_].next;  // unlink leaving edge

  nodes_[leave_child].parent_node = -1;
  nodes_[leave_child].parent_edge = -1;

  // Put leaving edge into NBV list in place of the entering edge
  is_bv_[leave_edge_] = false;
  nbv_[enter_nbv_pos_] = leave_edge_;

  // Add entering edge to BV as first child of its parent
  is_bv_[enter_eidx] = true;
  edges_[enter_eidx].flow = min_flow;
  std::ptrdiff_t enter_parent = edges_[enter_eidx].p;
  std::ptrdiff_t enter_child = edges_[enter_eidx].c;
  edges_[enter_eidx].next = nodes_[enter_parent].first_child;
  nodes_[enter_parent].first_child = enter_eidx;

  // Reverse parent-child path from enter_child to leave_child
  // (the subtree containing enter_child is now connected via the entering edge)
  std::ptrdiff_t prev_node = enter_parent;
  std::ptrdiff_t cur_node = enter_child;
  std::ptrdiff_t prev_edge = enter_eidx;

  while (cur_node >= 0) {
    std::ptrdiff_t next_node = nodes_[cur_node].parent_node;
    std::ptrdiff_t next_edge = nodes_[cur_node].parent_edge;

    // Set cur_node's new parent
    nodes_[cur_node].parent_node = prev_node;
    nodes_[cur_node].parent_edge = prev_edge;

    if (next_node >= 0) {
      // Remove next_edge from next_node's child list
      std::ptrdiff_t* ptr = &nodes_[next_node].first_child;
      while (*ptr != next_edge) {
        ptr = &edges_[*ptr].next;
      }
      *ptr = edges_[next_edge].next;

      // Reverse edge direction and prepend to cur_node's child list
      std::swap(edges_[next_edge].p, edges_[next_edge].c);
      edges_[next_edge].dir = 1 - edges_[next_edge].dir;
      edges_[next_edge].next = nodes_[cur_node].first_child;
      nodes_[cur_node].first_child = next_edge;
    }

    prev_edge = next_edge;
    prev_node = cur_node;
    cur_node = next_node;
  }

  // Update u and level for the subtree rooted at enter_child
  std::ptrdiff_t ep = edges_[enter_eidx].p;
  nodes_[enter_child].u =
      (edges_[enter_eidx].dir == 1) ? (nodes_[ep].u - 1) : (nodes_[ep].u + 1);
  nodes_[enter_child].level = nodes_[ep].level + 1;
}

inline double EmdSolver::total_flow() const {
  double total = 0.0;
  for (std::size_t e = 0; e < n_edges_; ++e) {
    if (is_bv_[e]) {
      total += edges_[e].flow;
    }
  }
  return total;
}

inline double EmdSolver::solve(std::ptrdiff_t root, int max_iter) {
  init_bv_tree(root);
  update_subtree(root);

  for (int iter = 0; iter < max_iter; ++iter) {
    if (is_optimal()) {
      break;
    }
    find_loop();
    // Save the entering child before pivot() replaces nbv_[enter_nbv_pos_].
    const std::ptrdiff_t enter_child = edges_[nbv_[enter_nbv_pos_]].c;
    pivot();
    // Only the reattached subtree needs its potentials refreshed.
    update_subtree(enter_child);
  }

  return total_flow();
}

}  // namespace detail

// ---------------------------------------------------------------------------
//  Public API: emd_l1
// ---------------------------------------------------------------------------

/// EMD-L1 for 1-D histograms — exact O(N) prefix-sum formula.
///
/// The Earth Mover's Distance under the L1 ground metric for 1-D discrete
/// histograms equals the sum of absolute values of the cumulative-sum
/// differences: EMD = Σ_k |Σ_{i≤k} (H1[i] − H2[i])|.
template <std::floating_point Scalar>
[[nodiscard]] Scalar emd_l1(const GridDataView<1, Scalar>& h1,
                             const GridDataView<1, Scalar>& h2) {
  if (h1.layout().shape() != h2.layout().shape()) {
    throw std::invalid_argument("histogram shapes do not match");
  }
  const std::size_t n = h1.layout().node_count();
  Scalar total{0};
  Scalar cumsum{0};
  for (std::size_t i = 0; i < n; ++i) {
    cumsum += h1.data()[i] - h2.data()[i];
    using std::abs;
    total += abs(cumsum);
  }
  return total;
}

/// EMD-L1 for multi-dimensional grid histograms using network simplex.
///
/// Implements the algorithm of Ling & Okada (PAMI 2007):
///   H. Ling and K. Okada,
///   "An Efficient Earth Mover's Distance Algorithm for Robust Histogram
///   Comparison," IEEE TPAMI 29(5):840–853, 2007.
///
/// The ground distance is the L1 (Manhattan) distance between bin indices.
/// Both histograms must share the same grid layout and have equal total mass.
template <std::size_t Dim, std::floating_point Scalar>
  requires(Dim >= 2)  // NOLINT(whitespace/indent_namespace)
[[nodiscard]] Scalar emd_l1(const GridDataView<Dim, Scalar>& h1,
                             const GridDataView<Dim, Scalar>& h2) {
  if (h1.layout().shape() != h2.layout().shape()) {
    throw std::invalid_argument("histogram shapes do not match");
  }
  const auto& layout = h1.layout();
  const auto& shape = layout.shape();
  const std::size_t n_nodes = layout.node_count();
  const std::size_t n_edges = layout.edge_count();

  detail::EmdSolver solver(n_nodes, n_edges);

  // Set node demands d[i] = H1[i] - H2[i]
  for (std::size_t i = 0; i < n_nodes; ++i) {
    solver.node(static_cast<std::ptrdiff_t>(i)).d =
        static_cast<double>(h1.data()[i]) -
        static_cast<double>(h2.data()[i]);
  }

  // ---- Greedy initial Basic Feasible Solution ----------------------------
  // Maintain working copy of demands (modified during greedy)
  std::vector<double> demand(n_nodes);
  for (std::size_t i = 0; i < n_nodes; ++i) {
    demand[i] = solver.node(static_cast<std::ptrdiff_t>(i)).d;
  }

  // prefix[a][k] = -(sum of demand[i] for all i with coord[a] < k),
  // maintained during the sweep.  Initialised from the original demands.
  std::vector<std::vector<double>> prefix(Dim);
  for (std::size_t a = 0; a < Dim; ++a) {
    prefix[a].assign(shape[a], 0.0);
    // Compute per-slice sums first
    std::vector<double> slice(shape[a], 0.0);
    for (std::size_t i = 0; i < n_nodes; ++i) {
      auto coord = layout.coordinates(static_cast<std::ptrdiff_t>(i));
      slice[static_cast<std::size_t>(coord[a])] += demand[i];
    }
    // Build prefix sums: prefix[a][k] = -(sum of slice[0..k-1])
    for (std::size_t k = 0; k + 1 < shape[a]; ++k) {
      prefix[a][k + 1] = prefix[a][k] - slice[k];
    }
  }

  // Strides: stride[a] = product of shape[a+1..Dim-1]
  std::array<std::ptrdiff_t, Dim> stride{};
  stride[Dim - 1] = 1;
  for (std::size_t a = Dim - 1; a-- > 0;) {
    stride[a] = stride[a + 1] * static_cast<std::ptrdiff_t>(shape[a + 1]);
  }

  // Sweep nodes 0 .. N-2 in lexicographic (flat) order
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n_nodes) - 1;
       ++i) {
    auto coord = layout.coordinates(i);
    const double d_i = demand[i];

    // Choose the axis that minimises |d_i + prefix[a][coord[a]+1]|
    std::size_t best_axis = Dim;  // sentinel
    double best_cost = std::numeric_limits<double>::max();
    for (std::size_t a = 0; a < Dim; ++a) {
      const std::size_t k = static_cast<std::size_t>(coord[a]);
      if (k + 1 < shape[a]) {
        using std::abs;
        const double cost = abs(d_i + prefix[a][k + 1]);
        if (cost < best_cost) {
          best_cost = cost;
          best_axis = a;
        }
      }
    }

    const std::size_t ka = static_cast<std::size_t>(coord[best_axis]);
    const std::ptrdiff_t neighbour = i + stride[best_axis];

    // BV edge: i → neighbour
    const int bv_dir = (d_i > 0.0) ? 1 : 0;
    using std::abs;
    solver.register_bv(i, neighbour, abs(d_i), bv_dir);

    // Update working arrays
    demand[static_cast<std::size_t>(neighbour)] += d_i;
    prefix[best_axis][ka + 1] += d_i;

    // All other forward edges from i are NBV
    for (std::size_t a = 0; a < Dim; ++a) {
      if (a == best_axis) continue;
      const std::size_t k = static_cast<std::size_t>(coord[a]);
      if (k + 1 < shape[a]) {
        solver.register_nbv(i, i + stride[a]);
      }
    }
  }

  // Choose the grid centre as the spanning-tree root
  std::ptrdiff_t root = 0;
  {
    typename GridLayout<Dim>::Coordinates rc{};
    for (std::size_t a = 0; a < Dim; ++a) {
      rc[a] = static_cast<std::ptrdiff_t>(shape[a] / 2);
      if (shape[a] > 1 && rc[a] > 0) --rc[a];
    }
    root = layout.node(rc);
  }

  return static_cast<Scalar>(solver.solve(root));
}

}  // namespace emdgrid
