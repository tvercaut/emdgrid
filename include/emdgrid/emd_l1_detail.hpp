#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace emdgrid {

struct SparseTransportPlan;

namespace detail {

struct DirectedEdgeFlow {
  std::ptrdiff_t from;
  std::ptrdiff_t to;
  double flow;
};

// ---------------------------------------------------------------------------
//  Data structures for the network-simplex EMD-L1 solver
// ---------------------------------------------------------------------------

struct LingOkadaGridNode {
  double d{0.0};                   // supply/demand = H1[i] - H2[i]
  int u{0};                        // dual variable (potential)
  int level{-1};                   // depth in BV tree (-1=unvisited)
  std::ptrdiff_t parent_node{-1};  // index of parent node  (-1 = root)
  std::ptrdiff_t parent_edge{-1};  // index of parent BV edge (-1 = root)
  std::ptrdiff_t first_child{-1};  // head of linked-list of child BV edges
};

struct LingOkadaGridEdge {
  std::ptrdiff_t p{-1};    // parent endpoint in BV tree orientation
  std::ptrdiff_t c{-1};    // child  endpoint
  double flow{0.0};
  int dir{1};              // 1 = outward (p→c), 0 = inward
  std::ptrdiff_t next{-1};  // next sibling in parent's child-edge linked list
};

// ---------------------------------------------------------------------------
//  Network-simplex solver
// ---------------------------------------------------------------------------
class LingOkadaSolver {
 public:
  LingOkadaSolver(std::size_t num_nodes, std::size_t num_edges)
      : m_nodes(num_nodes),
        m_edges(num_edges),
        m_is_bv(num_edges, false),
        m_aux(num_nodes + 4),
        m_from_loop(num_nodes + 4),
        m_to_loop(num_nodes + 4) {
    m_nbv.reserve(num_edges);
  }

  LingOkadaGridNode& node(std::ptrdiff_t i) { return m_nodes[i]; }

  // Register a BV edge chosen by the greedy step.
  // `from_node` is the processed node, `to_node` is its greedy neighbour.
  void register_bv(std::ptrdiff_t from_node, std::ptrdiff_t to_node,
                   double flow, int dir) {
    std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(m_n_edges++);
    m_edges[idx] = {from_node, to_node, flow, dir, -1};
    m_is_bv[idx] = true;
  }

  // Register a non-basic (NBV) edge from the greedy step.
  void register_nbv(std::ptrdiff_t low, std::ptrdiff_t high) {
    std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(m_n_edges++);
    m_edges[idx] = {low, high, 0.0, 1, -1};
    m_is_bv[idx] = false;
    m_nbv.push_back(idx);
  }

  // Run network simplex and return the EMD-L1 value.
  double solve(std::ptrdiff_t root, int max_iter = 500);

  [[nodiscard]] std::vector<DirectedEdgeFlow> get_directed_edge_flows() const {
    std::vector<DirectedEdgeFlow> flows;
    for (std::size_t e = 0; e < m_n_edges; ++e) {
      if (m_is_bv[e] && m_edges[e].flow > 0.0) {
        if (m_edges[e].dir == 1) {
          flows.push_back({m_edges[e].p, m_edges[e].c, m_edges[e].flow});
        } else {
          flows.push_back({m_edges[e].c, m_edges[e].p, m_edges[e].flow});
        }
      }
    }
    return flows;
  }

 private:
  void init_bv_tree(std::ptrdiff_t root);
  void update_subtree(std::ptrdiff_t start);
  bool is_optimal();
  void find_loop();
  void pivot();
  double total_flow() const;

  std::vector<LingOkadaGridNode> m_nodes;
  std::vector<LingOkadaGridEdge> m_edges;
  std::vector<bool> m_is_bv;
  std::vector<std::ptrdiff_t> m_nbv;
  std::size_t m_n_edges{0};

  // Pivot state
  std::ptrdiff_t m_enter_nbv_pos{-1};  // index into m_nbv[] of entering edge
  std::ptrdiff_t m_leave_edge{-1};     // edge index of leaving BV edge
  int m_leave_flag{0};                 // 0 = leave in from_loop, 1 = in to_loop

  std::vector<std::ptrdiff_t> m_aux;
  std::vector<std::ptrdiff_t> m_from_loop;
  std::vector<std::ptrdiff_t> m_to_loop;
  std::ptrdiff_t m_i_from{0};
  std::ptrdiff_t m_i_to{0};
};

// ---------------------------------------------------------------------------
//  Implementation
// ---------------------------------------------------------------------------

inline void LingOkadaSolver::init_bv_tree(std::ptrdiff_t root) {
  const auto n = static_cast<std::ptrdiff_t>(m_nodes.size());

  // Build BV adjacency: for each BV edge e, record it at both endpoints.
  std::vector<std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>>> adj(n);
  for (std::ptrdiff_t e = 0;
       e < static_cast<std::ptrdiff_t>(m_n_edges); ++e) {
    if (m_is_bv[e]) {
      adj[m_edges[e].p].emplace_back(e, m_edges[e].c);
      adj[m_edges[e].c].emplace_back(e, m_edges[e].p);
    }
  }

  m_nodes[root].level = 0;
  m_nodes[root].parent_node = -1;
  m_nodes[root].parent_edge = -1;
  m_nodes[root].first_child = -1;

  std::ptrdiff_t head = 0;
  std::ptrdiff_t tail = 0;
  m_aux[tail++] = root;

  while (head < tail) {
    std::ptrdiff_t u = m_aux[head++];
    for (auto [eidx, v] : adj[u]) {
      if (m_nodes[v].level == -1) {
        // v is not yet in the tree: make v a child of u via edge eidx
        m_nodes[v].level = m_nodes[u].level + 1;
        m_nodes[v].parent_node = u;
        m_nodes[v].parent_edge = eidx;
        m_nodes[v].first_child = -1;

        // Orient edge so that p == u, c == v
        if (m_edges[eidx].c == u) {
          std::swap(m_edges[eidx].p, m_edges[eidx].c);
          m_edges[eidx].dir = 1 - m_edges[eidx].dir;
        }

        // Prepend eidx to u's child list
        m_edges[eidx].next = m_nodes[u].first_child;
        m_nodes[u].first_child = eidx;

        m_aux[tail++] = v;
      }
    }
  }
}

inline void LingOkadaSolver::update_subtree(std::ptrdiff_t start) {
  std::ptrdiff_t head = 0;
  std::ptrdiff_t tail = 0;
  m_aux[tail++] = start;

  while (head < tail) {
    std::ptrdiff_t u = m_aux[head++];
    std::ptrdiff_t eidx = m_nodes[u].first_child;
    while (eidx >= 0) {
      std::ptrdiff_t v = m_edges[eidx].c;  // child node
      m_nodes[v].level = m_nodes[u].level + 1;
      // Outward edge (u→v): u's potential is one step above v.
      // Dual condition: c_{pq} = u[p] - u[q] = 1, so u[q] = u[p] - 1.
      m_nodes[v].u =
          (m_edges[eidx].dir == 1) ? (m_nodes[u].u - 1)
                                   : (m_nodes[u].u + 1);
      m_aux[tail++] = v;
      eidx = m_edges[eidx].next;
    }
  }
}

inline bool LingOkadaSolver::is_optimal() {
  m_enter_nbv_pos = -1;
  int min_cost = 0;

  for (std::size_t k = 0; k < m_nbv.size(); ++k) {
    std::ptrdiff_t eidx = m_nbv[k];
    const int cp = m_nodes[m_edges[eidx].p].u;
    const int cc = m_nodes[m_edges[eidx].c].u;
    // Reduced cost p→c: 1 - u[p] + u[c]
    const int cost = 1 - cp + cc;
    if (cost < min_cost) {
      min_cost = cost;
      m_enter_nbv_pos = static_cast<std::ptrdiff_t>(k);
    }
    // Reduced cost c→p: 1 + u[p] - u[c]
    const int cost_rev = 1 + cp - cc;
    if (cost_rev < min_cost) {
      min_cost = cost_rev;
      m_enter_nbv_pos = static_cast<std::ptrdiff_t>(k);
    }
  }

  if (m_enter_nbv_pos < 0) {
    return true;  // already optimal
  }

  // Orient the entering edge so that flow goes p→c (dir = 1).
  std::ptrdiff_t eidx = m_nbv[m_enter_nbv_pos];
  const int cp = m_nodes[m_edges[eidx].p].u;
  const int cc = m_nodes[m_edges[eidx].c].u;
  // If the reversed direction gave the minimum cost, swap endpoints.
  if (min_cost == (1 + cp - cc)) {
    std::swap(m_edges[eidx].p, m_edges[eidx].c);
  }
  m_edges[eidx].dir = 1;
  return false;
}

inline void LingOkadaSolver::find_loop() {
  // Entering edge: m_edges[m_nbv[m_enter_nbv_pos]]
  // m_from_loop: ancestors of entering parent;
  // m_to_loop:   ancestors of entering child.
  std::ptrdiff_t enter_eidx = m_nbv[m_enter_nbv_pos];
  std::ptrdiff_t from_node = m_edges[enter_eidx].p;  // entering parent
  std::ptrdiff_t to_node = m_edges[enter_eidx].c;    // entering child

  m_i_from = 0;
  m_i_to = 0;
  m_leave_edge = -1;
  m_leave_flag = 0;
  double min_flow = std::numeric_limits<double>::max();

  // Bring from_node and to_node to the same tree level
  while (m_nodes[from_node].level > m_nodes[to_node].level) {
    std::ptrdiff_t eidx = m_nodes[from_node].parent_edge;
    m_from_loop[m_i_from++] = eidx;
    // Inward edge on from_loop is a leaving candidate
    if (m_edges[eidx].dir == 0 && m_edges[eidx].flow < min_flow) {
      min_flow = m_edges[eidx].flow;
      m_leave_edge = eidx;
      m_leave_flag = 0;
    }
    from_node = m_nodes[from_node].parent_node;
  }

  while (m_nodes[to_node].level > m_nodes[from_node].level) {
    std::ptrdiff_t eidx = m_nodes[to_node].parent_edge;
    m_to_loop[m_i_to++] = eidx;
    // Outward edge on to_loop is a leaving candidate
    if (m_edges[eidx].dir == 1 && m_edges[eidx].flow < min_flow) {
      min_flow = m_edges[eidx].flow;
      m_leave_edge = eidx;
      m_leave_flag = 1;
    }
    to_node = m_nodes[to_node].parent_node;
  }

  // Advance both until LCA is found
  while (from_node != to_node) {
    {
      std::ptrdiff_t eidx = m_nodes[from_node].parent_edge;
      m_from_loop[m_i_from++] = eidx;
      if (m_edges[eidx].dir == 0 && m_edges[eidx].flow < min_flow) {
        min_flow = m_edges[eidx].flow;
        m_leave_edge = eidx;
        m_leave_flag = 0;
      }
      from_node = m_nodes[from_node].parent_node;
    }
    {
      std::ptrdiff_t eidx = m_nodes[to_node].parent_edge;
      m_to_loop[m_i_to++] = eidx;
      if (m_edges[eidx].dir == 1 && m_edges[eidx].flow < min_flow) {
        min_flow = m_edges[eidx].flow;
        m_leave_edge = eidx;
        m_leave_flag = 1;
      }
      to_node = m_nodes[to_node].parent_node;
    }
  }

  // If the leaving edge is on the from_loop side, reverse the entering edge.
  if (m_leave_flag == 0) {
    std::ptrdiff_t eidx = m_nbv[m_enter_nbv_pos];
    std::swap(m_edges[eidx].p, m_edges[eidx].c);
    m_edges[eidx].dir = 1 - m_edges[eidx].dir;
  }
}

inline void LingOkadaSolver::pivot() {
  std::ptrdiff_t enter_eidx = m_nbv[m_enter_nbv_pos];
  const double min_flow = m_edges[m_leave_edge].flow;

  // Update flows along the loop
  for (std::ptrdiff_t k = 0; k < m_i_from; ++k) {
    std::ptrdiff_t eidx = m_from_loop[k];
    if (m_edges[eidx].dir == 1) {
      m_edges[eidx].flow += min_flow;
    } else {
      m_edges[eidx].flow -= min_flow;
    }
  }
  for (std::ptrdiff_t k = 0; k < m_i_to; ++k) {
    std::ptrdiff_t eidx = m_to_loop[k];
    if (m_edges[eidx].dir == 1) {
      m_edges[eidx].flow -= min_flow;
    } else {
      m_edges[eidx].flow += min_flow;
    }
  }

  // Remove leaving BV edge from its parent's child list
  std::ptrdiff_t leave_parent = m_edges[m_leave_edge].p;
  std::ptrdiff_t leave_child = m_edges[m_leave_edge].c;

  std::ptrdiff_t* child_ptr = &m_nodes[leave_parent].first_child;
  while (*child_ptr != m_leave_edge) {
    child_ptr = &m_edges[*child_ptr].next;
  }
  *child_ptr = m_edges[m_leave_edge].next;  // unlink leaving edge

  m_nodes[leave_child].parent_node = -1;
  m_nodes[leave_child].parent_edge = -1;

  // Put leaving edge into NBV list in place of the entering edge
  m_is_bv[m_leave_edge] = false;
  m_nbv[m_enter_nbv_pos] = m_leave_edge;

  // Add entering edge to BV as first child of its parent
  m_is_bv[enter_eidx] = true;
  m_edges[enter_eidx].flow = min_flow;
  std::ptrdiff_t enter_parent = m_edges[enter_eidx].p;
  std::ptrdiff_t enter_child = m_edges[enter_eidx].c;
  m_edges[enter_eidx].next = m_nodes[enter_parent].first_child;
  m_nodes[enter_parent].first_child = enter_eidx;

  // Reverse parent-child path from enter_child to leave_child
  // (the subtree containing enter_child is now connected via entering edge)
  std::ptrdiff_t prev_node = enter_parent;
  std::ptrdiff_t cur_node = enter_child;
  std::ptrdiff_t prev_edge = enter_eidx;

  while (cur_node >= 0) {
    std::ptrdiff_t next_node = m_nodes[cur_node].parent_node;
    std::ptrdiff_t next_edge = m_nodes[cur_node].parent_edge;

    // Set cur_node's new parent
    m_nodes[cur_node].parent_node = prev_node;
    m_nodes[cur_node].parent_edge = prev_edge;

    if (next_node >= 0) {
      // Remove next_edge from next_node's child list
      std::ptrdiff_t* ptr = &m_nodes[next_node].first_child;
      while (*ptr != next_edge) {
        ptr = &m_edges[*ptr].next;
      }
      *ptr = m_edges[next_edge].next;

      // Reverse edge direction and prepend to cur_node's child list
      std::swap(m_edges[next_edge].p, m_edges[next_edge].c);
      m_edges[next_edge].dir = 1 - m_edges[next_edge].dir;
      m_edges[next_edge].next = m_nodes[cur_node].first_child;
      m_nodes[cur_node].first_child = next_edge;
    }

    prev_edge = next_edge;
    prev_node = cur_node;
    cur_node = next_node;
  }

  // Update u and level for the subtree rooted at enter_child
  std::ptrdiff_t ep = m_edges[enter_eidx].p;
  m_nodes[enter_child].u =
      (m_edges[enter_eidx].dir == 1) ? (m_nodes[ep].u - 1)
                                     : (m_nodes[ep].u + 1);
  m_nodes[enter_child].level = m_nodes[ep].level + 1;
}

inline double LingOkadaSolver::total_flow() const {
  double total = 0.0;
  for (std::size_t e = 0; e < m_n_edges; ++e) {
    if (m_is_bv[e]) {
      total += m_edges[e].flow;
    }
  }
  return total;
}

inline double LingOkadaSolver::solve(std::ptrdiff_t root, int max_iter) {
  init_bv_tree(root);
  update_subtree(root);

  for (int iter = 0; iter < max_iter; ++iter) {
    if (is_optimal()) {
      break;
    }
    find_loop();
    // Save the entering child before pivot() replaces m_nbv[m_enter_nbv_pos].
    const std::ptrdiff_t enter_child = m_edges[m_nbv[m_enter_nbv_pos]].c;
    pivot();
    // Only the reattached subtree needs its potentials refreshed.
    update_subtree(enter_child);
  }

  return total_flow();
}

struct SupplyItem {
  uint32_t source;
  double amount;
};

inline void extract_transport_plan(
    std::size_t n_nodes,
    const std::vector<double>& h1_data,
    const std::vector<double>& h2_data,
    const std::vector<DirectedEdgeFlow>& edge_flows,
    SparseTransportPlan* plan) {
  if (!plan) {
    return;
  }
  plan->source.clear();
  plan->target.clear();
  plan->flow.clear();

  std::vector<double> supply(n_nodes, 0.0);
  std::vector<double> demand(n_nodes, 0.0);

  for (std::size_t i = 0; i < n_nodes; ++i) {
    double self_mass = std::min(h1_data[i], h2_data[i]);
    if (self_mass > 0.0) {
      plan->source.push_back(static_cast<uint32_t>(i));
      plan->target.push_back(static_cast<uint32_t>(i));
      plan->flow.push_back(self_mass);
    }
    supply[i] = h1_data[i] - self_mass;
    demand[i] = h2_data[i] - self_mass;
  }

  // Build adjacency list for edge flows
  std::vector<std::vector<std::size_t>> out_edges(n_nodes);
  std::vector<std::size_t> in_degree(n_nodes, 0);

  for (std::size_t e = 0; e < edge_flows.size(); ++e) {
    std::size_t u = static_cast<std::size_t>(edge_flows[e].from);
    std::size_t v = static_cast<std::size_t>(edge_flows[e].to);
    out_edges[u].push_back(e);
    in_degree[v]++;
  }

  // Kahn's algorithm for topological ordering
  std::vector<std::size_t> zero_in_nodes;
  zero_in_nodes.reserve(n_nodes);
  for (std::size_t i = 0; i < n_nodes; ++i) {
    if (in_degree[i] == 0) {
      zero_in_nodes.push_back(i);
    }
  }

  std::vector<std::vector<SupplyItem>> node_supplies(n_nodes);
  for (std::size_t i = 0; i < n_nodes; ++i) {
    if (supply[i] > 0.0) {
      node_supplies[i].push_back({static_cast<uint32_t>(i), supply[i]});
    }
  }

  std::vector<std::size_t> current_in_degree = in_degree;
  std::size_t head = 0;
  while (head < zero_in_nodes.size()) {
    std::size_t u = zero_in_nodes[head++];

    // 1. Satisfy demand at u using node_supplies[u]
    double d_u = demand[u];
    while (d_u > 1e-12 && !node_supplies[u].empty()) {
      auto& item = node_supplies[u].back();
      double take = std::min(d_u, item.amount);
      plan->source.push_back(item.source);
      plan->target.push_back(static_cast<uint32_t>(u));
      plan->flow.push_back(take);

      d_u -= take;
      item.amount -= take;
      if (item.amount <= 1e-12) {
        node_supplies[u].pop_back();
      }
    }

    // 2. Forward remaining supplies at u along outgoing edge flows
    for (std::size_t e_idx : out_edges[u]) {
      const auto& edge = edge_flows[e_idx];
      std::size_t v = static_cast<std::size_t>(edge.to);
      double f_needed = edge.flow;

      while (f_needed > 1e-12 && !node_supplies[u].empty()) {
        auto& item = node_supplies[u].back();
        double take = std::min(f_needed, item.amount);
        node_supplies[v].push_back({item.source, take});

        f_needed -= take;
        item.amount -= take;
        if (item.amount <= 1e-12) {
          node_supplies[u].pop_back();
        }
      }

      current_in_degree[v]--;
      if (current_in_degree[v] == 0) {
        zero_in_nodes.push_back(v);
      }
    }
  }
}

}  // namespace detail

}  // namespace emdgrid
