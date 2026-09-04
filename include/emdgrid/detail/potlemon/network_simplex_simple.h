/* -*- mode: C++; indent-tabs-mode: nil; -*-
 *
 *
 * This file has been adapted by Nicolas Bonneel (2013),
 * from network_simplex.h from LEMON, a generic C++ optimization library,
 * to implement a lightweight network simplex for mass transport, more
 * memory efficient that the original file. A previous version of this file
 * is used as part of the Displacement Interpolation project,
 * Web: http://www.cs.ubc.ca/labs/imager/tr/2011/DisplacementInterpolation/
 *
 *
 **** Original file Copyright Notice :
 *
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
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <queue>
#include <unordered_map>
#include <stack>
#include <utility>
#include <vector>

#include "emdgrid/detail/potlemon/sparse_bipartitegraph.h"

#pragma push_macro("MAX")
#pragma push_macro("MIN")
#pragma push_macro("INVALID")
#pragma push_macro("INVALIDNODE")

#undef MAX
#undef MIN
#undef INVALIDNODE
#undef INVALID

#define INVALIDNODE (-1)
#define INVALID (-1)

#ifndef EPSILON
#define EPSILON 2.2204460492503131e-15
#endif

#ifndef _EPSILON
#define _EPSILON 1e-8
#endif

namespace potlemon {

/// Type alias for hash map used in Network Simplex sparse structures.
template <typename Key, typename Value>
using HashMap = std::unordered_map<Key, Value>;

template <typename T>
class ProxyObject;

template <typename T>
class SparseValueVector {
 public:
  explicit SparseValueVector(size_t n = 0) { (void)n; }
  void resize(size_t n = 0) { (void)n; }
  T operator[](const size_t id) const {
    typename HashMap<size_t, T>::const_iterator it = data.find(id);
    if (it == data.end()) {
      return 0;
    }
    return it->second;
  }

  ProxyObject<T> operator[](const size_t id) {
    return ProxyObject<T>(this, id);
  }

  HashMap<size_t, T> data;
};

template <typename T>
class ProxyObject {
 public:
  ProxyObject(SparseValueVector<T>* v, size_t idx) : m_v(v), m_idx(idx) {}
  ProxyObject<T>& operator=(const T& v) {
    if (v != 0) {
      m_v->data[m_idx] = v;
    }
    return *this;
  }

  operator T() {  // NOLINT(google-explicit-constructor)
    typename HashMap<size_t, T>::iterator it = m_v->data.find(m_idx);
    if (it == m_v->data.end()) {
      return 0;
    }
    return it->second;
  }

  void operator+=(T val) {
    if (val == 0) {
      return;
    }
    typename HashMap<size_t, T>::iterator it = m_v->data.find(m_idx);
    if (it == m_v->data.end()) {
      m_v->data[m_idx] = val;
    } else {
      T sum = it->second + val;
      if (sum == 0) {
        m_v->data.erase(it);
      } else {
        it->second = sum;
      }
    }
  }

  void operator-=(T val) {
    if (val == 0) {
      return;
    }
    typename HashMap<size_t, T>::iterator it = m_v->data.find(m_idx);
    if (it == m_v->data.end()) {
      m_v->data[m_idx] = -val;
    } else {
      T sum = it->second - val;
      if (sum == 0) {
        m_v->data.erase(it);
      } else {
        it->second = sum;
      }
    }
  }

  SparseValueVector<T>* m_v;
  size_t m_idx;
};

template <typename GR, typename V = int64_t, typename C = V,
          typename NodesType = uint16_t, typename ArcsType = int64_t>
class NetworkSimplexSimple {  // NOLINT(whitespace/indent_namespace)
 public:
  enum class CostMode { StoredArray, DenseMatrix, LazyGeometry };

  enum class CostStorageMode { AllArcCosts, ArtificialArcCosts };

  enum class FlowStorageMode { Dense, SparseArcFlows };

  enum class EndpointStorageMode { Dense, ArcEndpoints };

  enum class StateStorageMode { Dense, PackedArcStates };

  struct SimplexOptions {
    bool arc_mixing;
    CostStorageMode cost_storage_mode;
    FlowStorageMode flow_storage_mode;
    EndpointStorageMode endpoint_storage_mode;
    StateStorageMode state_storage_mode;

    explicit SimplexOptions(bool arc_mixing_ = false)
        : arc_mixing(arc_mixing_),
          cost_storage_mode(CostStorageMode::AllArcCosts),
          flow_storage_mode(FlowStorageMode::Dense),
          endpoint_storage_mode(EndpointStorageMode::Dense),
          state_storage_mode(StateStorageMode::Dense) {}
  };

  NetworkSimplexSimple(const GR& graph, SimplexOptions options, int nbnodes,
                       ArcsType nb_arcs, uint64_t maxiters)
      : _graph(graph),
        _n1(0),
        _n2(0),
        _max_cost(0),
        _has_max_cost(false),
        _arc_mixing(options.arc_mixing),
        _cost_mode(CostMode::StoredArray),
        _cost_storage_mode(options.cost_storage_mode),
        _flow_storage_mode(options.flow_storage_mode),
        _endpoint_storage_mode(options.endpoint_storage_mode),
        _state_storage_mode(options.state_storage_mode),
        _coords_a(nullptr),
        _coords_b(nullptr),
        _dim(0),
        _metric(0),
        _D_ptr(nullptr),
        _D_n2(0),
        _warmstart_provided(false),
        _warmstart_tree_built(false),
        MAX(std::numeric_limits<Value>::max()),
        INF(std::numeric_limits<Value>::has_infinity
                ? std::numeric_limits<Value>::infinity()
                : MAX),
        _init_nb_nodes(nbnodes),
        _init_nb_arcs(nb_arcs) {
    reset();
    max_iter = maxiters;
  }

  typedef V Value;
  typedef C Cost;

  enum ProblemType { INFEASIBLE, OPTIMAL, UNBOUNDED, MAX_ITER_REACHED };

  enum SupplyType { GEQ, LEQ };

 private:
  uint64_t max_iter;
  POTLEMON_TEMPLATE_DIGRAPH_TYPEDEFS(GR)

  typedef std::vector<int> IntVector;
  typedef std::vector<ArcsType> ArcVector;
  typedef std::vector<Value> ValueVector;
  typedef std::vector<Cost> CostVector;
  typedef std::vector<char> BoolVector;

  enum ArcState { STATE_UPPER = -1, STATE_TREE = 0, STATE_LOWER = 1 };

  class PackedStateVector {
   public:
    void resize(ArcsType n) {
      _size = n;
      _data.assign((static_cast<size_t>(n) + 3) / 4, 0);
    }

    void clear() {
      _size = 0;
      _data.clear();
    }

    void fill(ArcsType count, signed char state) {
      for (ArcsType i = 0; i < count; ++i) {
        set(i, state);
      }
    }

    signed char get(ArcsType index) const {
      const uint8_t bits = (_data[static_cast<size_t>(index) / 4] >>
                            (2 * (static_cast<size_t>(index) % 4))) &
                           0x03;
      if (bits == 0) return STATE_LOWER;
      if (bits == 1) return STATE_TREE;
      return STATE_UPPER;
    }

    void set(ArcsType index, signed char state) {
      const size_t byte_index = static_cast<size_t>(index) / 4;
      const size_t shift = 2 * (static_cast<size_t>(index) % 4);
      _data[byte_index] = static_cast<uint8_t>(
          (_data[byte_index] & ~(uint8_t(0x03) << shift)) |
          (encode(state) << shift));
    }

   private:
    static uint8_t encode(signed char state) {
      if (state == STATE_LOWER) return 0;
      if (state == STATE_TREE) return 1;
      return 2;
    }

    ArcsType _size;
    std::vector<uint8_t> _data;
  };

  typedef std::vector<signed char> StateVector;

 private:
  const GR& _graph;
  int _node_num;
  int _n1;
  int _n2;
  ArcsType _arc_num;
  ArcsType _all_arc_num;
  ArcsType _search_arc_num;

  SupplyType _stype;
  Value _sum_supply;
  Cost _max_cost;
  bool _has_max_cost;

  inline int _node_id(int n) const { return _node_num - n - 1; }

  IntVector _source;
  IntVector _target;
  IntVector _artificial_source;
  IntVector _artificial_target;
  bool _arc_mixing;

 public:
  CostVector _cost;
  ValueVector _supply;
  ValueVector _flow;
  ValueVector _artificial_flow;
  CostVector _pi;

  CostMode _cost_mode;
  CostStorageMode _cost_storage_mode;
  FlowStorageMode _flow_storage_mode;
  EndpointStorageMode _endpoint_storage_mode;
  StateStorageMode _state_storage_mode;
  const double* _coords_a;
  const double* _coords_b;
  int _dim;
  int _metric;

  const double* _D_ptr;
  int _D_n2;

 private:
  HashMap<ArcsType, Value> _real_flow;

  bool _warmstart_provided;
  bool _warmstart_tree_built;

  IntVector _parent;
  ArcVector _pred;
  IntVector _thread;
  IntVector _rev_thread;
  IntVector _succ_num;
  IntVector _last_succ;
  IntVector _dirty_revs;
  BoolVector _forward;
  StateVector _state;
  PackedStateVector _packed_state;
  ArcsType _root;

  ArcsType in_arc, join, u_in, v_in, u_out, v_out;
  ArcsType first, second, right, last;
  ArcsType stem, par_stem, new_stem;
  Value delta;

  const Value MAX;
  ArcsType mixingCoeff;

 public:
  const Value INF;

 private:
  inline ArcsType sequence(ArcsType k) const {
    ArcsType smallv = (k > num_total_big_subsequence_numbers) & 1;
    k -= num_total_big_subsequence_numbers * smallv;
    ArcsType subsequence_length2 = subsequence_length - smallv;
    ArcsType subsequence_num =
        (k / subsequence_length2) + num_big_subseqiences * smallv;
    ArcsType subsequence_offset = (k % subsequence_length2) * mixingCoeff;
    return subsequence_offset + subsequence_num;
  }
  ArcsType subsequence_length;
  ArcsType num_big_subseqiences;
  ArcsType num_total_big_subsequence_numbers;

  inline ArcsType getArcID(const Arc& arc) const {
    ArcsType n = _arc_num - GR::id(arc) - 1;
    if (_arc_mixing)
      return sequence(n);
    else
      return n;
  }

  inline bool usesStoredCost() const {
    return _cost_mode == CostMode::StoredArray;
  }

  inline bool usesDenseCost() const {
    return _cost_mode == CostMode::DenseMatrix;
  }

  inline bool usesLazyCost() const {
    return _cost_mode == CostMode::LazyGeometry;
  }

  inline bool storesArtificialArcCosts() const {
    return _cost_storage_mode == CostStorageMode::ArtificialArcCosts;
  }

  inline bool storesSparseArcFlows() const {
    return _flow_storage_mode == FlowStorageMode::SparseArcFlows;
  }

  inline bool usesArcEndpoints() const {
    return _endpoint_storage_mode == EndpointStorageMode::ArcEndpoints;
  }

  inline bool usesPackedArcStates() const {
    return _state_storage_mode == StateStorageMode::PackedArcStates;
  }

  Cost computeLazyCostUpperBound() const {
    Cost squared_range_sum = 0;
    Cost l1_range_sum = 0;

    for (int d = 0; d < _dim; ++d) {
      Cost min_value = _coords_a[d];
      Cost max_value = _coords_a[d];

      for (int i = 0; i < _n1; ++i) {
        const Cost value = _coords_a[i * _dim + d];
        if (value < min_value) min_value = value;
        if (value > max_value) max_value = value;
      }
      for (int j = 0; j < _n2; ++j) {
        const Cost value = _coords_b[j * _dim + d];
        if (value < min_value) min_value = value;
        if (value > max_value) max_value = value;
      }

      const Cost range = max_value - min_value;
      squared_range_sum += range * range;
      l1_range_sum += range;
    }

    if (_metric == 0) return squared_range_sum;
    if (_metric == 1) return std::sqrt(squared_range_sum);
    return l1_range_sum;
  }

  Cost maxRealArcCost() {
    if (_has_max_cost) {
      return _max_cost;
    }

    Cost max_cost = 0;
    for (ArcsType i = 0; i != _arc_num; ++i) {
      Cost cost = getCostForArc(i);
      if (i == 0 || cost > max_cost) {
        max_cost = cost;
      }
    }
    _max_cost = max_cost;
    _has_max_cost = true;
    return max_cost;
  }

  inline int arcSource(ArcsType arc_id) const {
    if (usesArcEndpoints()) {
      if (arc_id < _arc_num) {
        const ArcsType graph_arc = _arc_num - arc_id - 1;
        return _node_id(static_cast<int>(graph_arc / _n2));
      }
      return _artificial_source[arc_id - _arc_num];
    }
    return _source[arc_id];
  }

  inline int arcTarget(ArcsType arc_id) const {
    if (usesArcEndpoints()) {
      if (arc_id < _arc_num) {
        const ArcsType graph_arc = _arc_num - arc_id - 1;
        return _node_id(static_cast<int>(graph_arc % _n2) + _n1);
      }
      return _artificial_target[arc_id - _arc_num];
    }
    return _target[arc_id];
  }

  inline void setArcEndpoints(ArcsType arc_id, int source, int target) {
    if (usesArcEndpoints()) {
      if (arc_id >= _arc_num) {
        _artificial_source[arc_id - _arc_num] = source;
        _artificial_target[arc_id - _arc_num] = target;
      }
      return;
    }
    _source[arc_id] = source;
    _target[arc_id] = target;
  }

  inline void setArcCost(ArcsType arc_id, Cost cost) {
    if (storesArtificialArcCosts()) {
      if (arc_id >= _arc_num) {
        _cost[arc_id - _arc_num] = cost;
      }
    } else {
      _cost[arc_id] = cost;
    }
    if (arc_id < _arc_num && (!_has_max_cost || cost > _max_cost)) {
      _max_cost = cost;
      _has_max_cost = true;
    }
  }

  inline signed char arcState(ArcsType arc_id) const {
    if (usesPackedArcStates()) {
      return _packed_state.get(arc_id);
    }
    return _state[arc_id];
  }

  inline void setArcState(ArcsType arc_id, signed char state) {
    if (usesPackedArcStates()) {
      _packed_state.set(arc_id, state);
      return;
    }
    _state[arc_id] = state;
  }

  inline void flipArcState(ArcsType arc_id) {
    setArcState(arc_id, -arcState(arc_id));
  }

  inline void fillArcStates(ArcsType count, signed char state) {
    if (usesPackedArcStates()) {
      _packed_state.fill(count, state);
    } else {
      std::fill_n(_state.begin(), count, state);
    }
  }

  inline ArcsType flowArcCount() const {
    return storesSparseArcFlows() ? _all_arc_num
                                  : static_cast<ArcsType>(_flow.size());
  }

  inline Value arcFlow(ArcsType arc_id) const {
    if (storesSparseArcFlows()) {
      if (arc_id < _arc_num) {
        typename HashMap<ArcsType, Value>::const_iterator it =
            _real_flow.find(arc_id);
        return it == _real_flow.end() ? Value(0) : it->second;
      }
      return _artificial_flow[arc_id - _arc_num];
    }
    return _flow[arc_id];
  }

  inline void setArcFlow(ArcsType arc_id, Value flow) {
    if (storesSparseArcFlows()) {
      if (arc_id < _arc_num) {
        if (flow == 0) {
          _real_flow.erase(arc_id);
        } else {
          _real_flow[arc_id] = flow;
        }
      } else {
        _artificial_flow[arc_id - _arc_num] = flow;
      }
      return;
    }
    _flow[arc_id] = flow;
  }

  inline void addArcFlow(ArcsType arc_id, Value delta) {
    if (storesSparseArcFlows()) {
      if (arc_id < _arc_num) {
        setArcFlow(arc_id, arcFlow(arc_id) + delta);
      } else {
        _artificial_flow[arc_id - _arc_num] += delta;
      }
      return;
    }
    _flow[arc_id] += delta;
  }

  class BlockSearchPivotRule {
   private:
    const CostVector& _pi;
    ArcsType& _in_arc;
    ArcsType _search_arc_num;
    ArcsType _block_size;
    ArcsType _next_arc;
    NetworkSimplexSimple& _ns;

   public:
    explicit BlockSearchPivotRule(NetworkSimplexSimple& ns)
        : _pi(ns._pi),
          _in_arc(ns.in_arc),
          _search_arc_num(ns._search_arc_num),
          _next_arc(0),
          _ns(ns) {
      const double BLOCK_SIZE_FACTOR = 1.0;
      const ArcsType MIN_BLOCK_SIZE = 10;
      _block_size = std::max(
          static_cast<ArcsType>(BLOCK_SIZE_FACTOR *
                                std::sqrt(static_cast<double>(
                                    _search_arc_num))),
          MIN_BLOCK_SIZE);
    }

    inline Cost getCost(ArcsType e) const { return _ns.getCostForArc(e); }

    bool findEnteringArc() {
      Cost c, min = 0;
      ArcsType e;
      ArcsType cnt = _block_size;
      double a;
      for (e = _next_arc; e != _search_arc_num; ++e) {
        c = _ns.arcState(e) *
            (getCost(e) + _pi[_ns.arcSource(e)] - _pi[_ns.arcTarget(e)]);
        if (c < min) {
          min = c;
          _in_arc = e;
        }
        if (--cnt == 0) {
          a = std::abs(_pi[_ns.arcSource(_in_arc)]) >
                      std::abs(_pi[_ns.arcTarget(_in_arc)])
                  ? std::abs(_pi[_ns.arcSource(_in_arc)])
                  : std::abs(_pi[_ns.arcTarget(_in_arc)]);
          a = a > std::abs(getCost(_in_arc)) ? a : std::abs(getCost(_in_arc));
          if (min < -EPSILON * a) goto search_end;
          cnt = _block_size;
        }
      }
      for (e = 0; e != _next_arc; ++e) {
        c = _ns.arcState(e) *
            (getCost(e) + _pi[_ns.arcSource(e)] - _pi[_ns.arcTarget(e)]);
        if (c < min) {
          min = c;
          _in_arc = e;
        }
        if (--cnt == 0) {
          a = std::abs(_pi[_ns.arcSource(_in_arc)]) >
                      std::abs(_pi[_ns.arcTarget(_in_arc)])
                  ? std::abs(_pi[_ns.arcSource(_in_arc)])
                  : std::abs(_pi[_ns.arcTarget(_in_arc)]);
          a = a > std::abs(getCost(_in_arc)) ? a : std::abs(getCost(_in_arc));
          if (min < -EPSILON * a) goto search_end;
          cnt = _block_size;
        }
      }
      a = std::abs(_pi[_ns.arcSource(_in_arc)]) >
                  std::abs(_pi[_ns.arcTarget(_in_arc)])
              ? std::abs(_pi[_ns.arcSource(_in_arc)])
              : std::abs(_pi[_ns.arcTarget(_in_arc)]);
      a = a > std::abs(getCost(_in_arc)) ? a : std::abs(getCost(_in_arc));
      if (min >= -EPSILON * a) return false;

    search_end:
      _next_arc = e;
      return true;
    }
  };

 public:
  ArcsType arcNum() const { return _arc_num; }
  int nodeNum() const { return _node_num; }
  int n1() const { return _n1; }
  int n2() const { return _n2; }
  Cost pi(int internal_node) const { return _pi[internal_node]; }

  int _init_nb_nodes;
  ArcsType _init_nb_arcs;

  template <typename CostMap>
  NetworkSimplexSimple& costMap(const CostMap& map) {
    Arc a;
    _graph.first(a);
    for (; a != INVALID; _graph.next(a)) {
      setArcCost(getArcID(a), map[a]);
    }
    return *this;
  }

  template <typename ValueType>
  NetworkSimplexSimple& setCost(const Arc& arc, const ValueType cost) {
    setArcCost(getArcID(arc), static_cast<Cost>(cost));
    return *this;
  }

  NetworkSimplexSimple& setLazyCost(const double* coords_a,
                                    const double* coords_b, int dim, int metric,
                                    int n1, int n2) {
    _cost_mode = CostMode::LazyGeometry;
    _coords_a = coords_a;
    _coords_b = coords_b;
    _dim = dim;
    _metric = metric;
    _n1 = n1;
    _n2 = n2;
    _max_cost = computeLazyCostUpperBound();
    _has_max_cost = true;
    return *this;
  }

  NetworkSimplexSimple& setDenseCostMatrix(const double* D, int n2) {
    _cost_mode = CostMode::DenseMatrix;
    _D_ptr = D;
    _D_n2 = n2;
    _has_max_cost = true;
    _max_cost = static_cast<Cost>(D[0]);
    for (ArcsType i = 1; i != _arc_num; ++i) {
      if (D[i] > _max_cost) _max_cost = static_cast<Cost>(D[i]);
    }
    return *this;
  }

  inline Cost computeLazyCost(int i, int j_adjusted) const {
    const double* xa = _coords_a + i * _dim;
    const double* xb = _coords_b + j_adjusted * _dim;
    Cost cost = 0;

    if (_metric == 0) {
      for (int d = 0; d < _dim; ++d) {
        Cost diff = static_cast<Cost>(xa[d] - xb[d]);
        cost += diff * diff;
      }
      return cost;
    } else if (_metric == 1) {
      for (int d = 0; d < _dim; ++d) {
        Cost diff = static_cast<Cost>(xa[d] - xb[d]);
        cost += diff * diff;
      }
      return std::sqrt(cost);
    } else {
      for (int d = 0; d < _dim; ++d) {
        cost += std::abs(static_cast<Cost>(xa[d] - xb[d]));
      }
      return cost;
    }
  }

  inline Cost getCostForArc(ArcsType arc_id) const {
    if (usesDenseCost()) {
      if (arc_id >= _arc_num) {
        return storesArtificialArcCosts() ? _cost[arc_id - _arc_num]
                                          : _cost[arc_id];
      }
      return static_cast<Cost>(_D_ptr[_arc_num - arc_id - 1]);
    } else if (usesStoredCost()) {
      return storesArtificialArcCosts() ? _cost[arc_id - _arc_num]
                                        : _cost[arc_id];
    } else {
      if (arc_id >= _arc_num) {
        return storesArtificialArcCosts() ? _cost[arc_id - _arc_num]
                                          : _cost[arc_id];
      }
      int i = _node_num - arcSource(arc_id) - 1;
      int j = _node_num - arcTarget(arc_id) - 1 - _n1;
      return computeLazyCost(i, j);
    }
  }

  template <typename SupplyMap>
  NetworkSimplexSimple& supplyMap(const SupplyMap& map) {
    Node n;
    _graph.first(n);
    for (; n != INVALIDNODE; _graph.next(n)) {
      _supply[_node_id(n)] = map[n];
    }
    return *this;
  }

  template <typename SupplyMap>
  NetworkSimplexSimple& supplyMap(const SupplyMap* map1, int n1,
                                  const SupplyMap* map2, int n2) {
    (void)n2;
    Node n;
    _graph.first(n);
    for (; n != INVALIDNODE; _graph.next(n)) {
      if (n < n1)
        _supply[_node_id(n)] = map1[n];
      else
        _supply[_node_id(n)] = map2[n - n1];
    }
    return *this;
  }

  template <typename SupplyMap>
  NetworkSimplexSimple& supplyMapAll(SupplyMap val1, int n1, SupplyMap val2,
                                     int n2) {
    Node n;
    _graph.first(n);
    for (; n != INVALIDNODE; _graph.next(n)) {
      if (n < n1)
        _supply[_node_id(n)] = val1;
      else
        _supply[_node_id(n)] = val2;
    }
    return *this;
  }

  NetworkSimplexSimple& stSupply(const Node& s, const Node& t, Value k) {
    for (int i = 0; i != _node_num; ++i) {
      _supply[i] = 0;
    }
    _supply[_node_id(s)] = k;
    _supply[_node_id(t)] = -k;
    return *this;
  }

  NetworkSimplexSimple& supplyType(SupplyType supply_type) {
    _stype = supply_type;
    return *this;
  }

  void setWarmstartPotentials(const Cost* alpha, const Cost* beta, int n,
                              int m) {
    for (int i = 0; i < n; ++i) {
      _pi[_node_id(i)] = -alpha[i];
    }
    for (int j = 0; j < m; ++j) {
      _pi[_node_id(n + j)] = beta[j];
    }
    _warmstart_provided = true;
  }

  ProblemType run() {
    if (_warmstart_provided) {
      if (!warmstartInit()) return INFEASIBLE;
      _warmstart_tree_built = true;
    } else {
      if (!init()) return INFEASIBLE;
      _warmstart_tree_built = false;
    }

    return start();
  }

  NetworkSimplexSimple& resetParams() {
    std::fill_n(_supply.begin(), _node_num, Value(0));
    if (usesStoredCost() && !storesArtificialArcCosts()) {
      std::fill_n(_cost.begin(), _arc_num, Cost(1));
    }
    _stype = GEQ;
    _has_max_cost = false;
    _warmstart_provided = false;
    _warmstart_tree_built = false;
    return *this;
  }

  NetworkSimplexSimple& reset() {
    _node_num = _init_nb_nodes;
    _arc_num = _init_nb_arcs;
    int all_node_num = _node_num + 1;
    ArcsType max_arc_num = _arc_num + 2 * _node_num;
    _all_arc_num = max_arc_num;

    if (usesArcEndpoints()) {
      _source.clear();
      _target.clear();
      _artificial_source.resize(2 * _node_num);
      _artificial_target.resize(2 * _node_num);
    } else {
      _source.resize(max_arc_num);
      _target.resize(max_arc_num);
      _artificial_source.clear();
      _artificial_target.clear();
    }

    if (storesArtificialArcCosts()) {
      _cost.resize(2 * _node_num);
    } else {
      _cost.resize(max_arc_num);
    }
    _supply.resize(all_node_num);
    if (storesSparseArcFlows()) {
      _flow.clear();
      _artificial_flow.assign(2 * _node_num, Value(0));
      _real_flow.clear();
    } else {
      _flow.resize(max_arc_num);
      _artificial_flow.clear();
      _real_flow.clear();
    }
    _pi.resize(all_node_num);

    _parent.resize(all_node_num);
    _pred.resize(all_node_num);
    _forward.resize(all_node_num);
    _thread.resize(all_node_num);
    _rev_thread.resize(all_node_num);
    _succ_num.resize(all_node_num);
    _last_succ.resize(all_node_num);
    if (usesPackedArcStates()) {
      _state.clear();
      _packed_state.resize(max_arc_num);
    } else {
      _state.resize(max_arc_num);
      _packed_state.clear();
    }

    if (usesArcEndpoints()) {
    } else if (_arc_mixing) {
      const ArcsType k = std::max(
          static_cast<ArcsType>(std::sqrt(static_cast<double>(_arc_num))),
          static_cast<ArcsType>(10));
      mixingCoeff = k;
      subsequence_length = _arc_num / mixingCoeff + 1;
      num_big_subseqiences = _arc_num % mixingCoeff;
      num_total_big_subsequence_numbers =
          subsequence_length * num_big_subseqiences;

      ArcsType i = 0, j = 0;
      Arc a;
      _graph.first(a);
      for (; a != INVALID; _graph.next(a)) {
        setArcEndpoints(i, _node_id(_graph.source(a)),
                        _node_id(_graph.target(a)));
        if ((i += k) >= _arc_num) i = ++j;
      }
    } else {
      ArcsType i = 0;
      Arc a;
      _graph.first(a);
      for (; a != INVALID; _graph.next(a), ++i) {
        setArcEndpoints(i, _node_id(_graph.source(a)),
                        _node_id(_graph.target(a)));
      }
    }

    resetParams();
    return *this;
  }

  template <typename Number>
  Number totalCost() const {
    Number c = 0;
    for (ArcsType i = 0; i < flowArcCount(); i++) {
      if (arcFlow(i) != 0) {
        c += static_cast<Number>(arcFlow(i)) *
             static_cast<Number>(getCostForArc(i));
      }
    }
    return c;
  }

  Cost totalCost() const { return totalCost<Cost>(); }

  Value flow(const Arc& a) const { return arcFlow(getArcID(a)); }

  template <typename FlowMap>
  void flowMap(FlowMap& map) const {
    Arc a;
    _graph.first(a);
    for (; a != INVALID; _graph.next(a)) {
      map.set(a, arcFlow(getArcID(a)));
    }
  }

  Cost potential(const Node& n) const { return _pi[_node_id(n)]; }

  template <typename PotentialMap>
  void potentialMap(PotentialMap& map) const {
    Node n;
    _graph.first(n);
    for (; n != INVALIDNODE; _graph.next(n)) {
      map.set(n, _pi[_node_id(n)]);
    }
  }

 private:
  bool warmstartInit() {
    if (_node_num == 0) return false;

    _sum_supply = 0;
    for (int i = 0; i != _node_num; ++i) {
      _sum_supply += _supply[i];
    }
    if (std::abs(static_cast<double>(_sum_supply)) > _EPSILON) return false;
    _sum_supply = 0;
    int tree_edges = 0;
    std::vector<ArcsType> tree_arcs;
    tree_arcs.reserve(_node_num);
    Cost ART_COST = 0;

    {
      ArcsType K = std::min(static_cast<ArcsType>(4 * _node_num), _arc_num);
      typedef std::pair<Cost, ArcsType> HeapEntry;
      std::priority_queue<HeapEntry> maxheap;

      for (ArcsType e = 0; e < _arc_num; ++e) {
        setArcState(e, STATE_LOWER);
        Cost c = getCostForArc(e);
        if (c > ART_COST) ART_COST = c;
        Cost rc = std::abs(c + _pi[arcSource(e)] - _pi[arcTarget(e)]);
        if (static_cast<ArcsType>(maxheap.size()) < K) {
          maxheap.push({rc, e});
        } else if (rc < maxheap.top().first) {
          maxheap.pop();
          maxheap.push({rc, e});
        }
      }
      if (std::numeric_limits<Cost>::is_exact) {
        ART_COST = std::numeric_limits<Cost>::max() / 2 + 1;
      } else {
        ART_COST = (ART_COST + 1) * _node_num;
      }

      std::vector<HeapEntry> candidates;
      candidates.reserve(maxheap.size());
      while (!maxheap.empty()) {
        candidates.push_back(maxheap.top());
        maxheap.pop();
      }

      std::sort(candidates.begin(), candidates.end(),
                [](const HeapEntry& a, const HeapEntry& b) {
                  return a.first < b.first;
                });

      std::vector<int> uf_parent(_node_num);
      std::vector<int> uf_rank(_node_num, 0);
      for (int i = 0; i < _node_num; ++i) uf_parent[i] = i;

      for (ArcsType idx = 0; idx < static_cast<ArcsType>(candidates.size()) &&
                             tree_edges < _node_num - 1;
           ++idx) {
        ArcsType e = candidates[idx].second;
        int s = arcSource(e);
        int t = arcTarget(e);
        int rs = s, rt = t;
        while (uf_parent[rs] != rs) {
          uf_parent[rs] = uf_parent[uf_parent[rs]];
          rs = uf_parent[rs];
        }
        while (uf_parent[rt] != rt) {
          uf_parent[rt] = uf_parent[uf_parent[rt]];
          rt = uf_parent[rt];
        }
        if (rs == rt) continue;
        if (uf_rank[rs] < uf_rank[rt]) std::swap(rs, rt);
        uf_parent[rt] = rs;
        if (uf_rank[rs] == uf_rank[rt]) uf_rank[rs]++;
        tree_arcs.push_back(e);
        tree_edges++;
      }

      if (tree_edges < _node_num - 1) {
        std::vector<bool> considered(_arc_num, false);
        for (auto& c : candidates) considered[c.second] = true;

        for (ArcsType e = 0;
             e < _arc_num && tree_edges < _node_num - 1; ++e) {
          if (considered[e]) continue;
          int s = arcSource(e);
          int t = arcTarget(e);
          int rs = s, rt = t;
          while (uf_parent[rs] != rs) {
            uf_parent[rs] = uf_parent[uf_parent[rs]];
            rs = uf_parent[rs];
          }
          while (uf_parent[rt] != rt) {
            uf_parent[rt] = uf_parent[uf_parent[rt]];
            rt = uf_parent[rt];
          }
          if (rs == rt) continue;
          if (uf_rank[rs] < uf_rank[rt]) std::swap(rs, rt);
          uf_parent[rt] = rs;
          if (uf_rank[rs] == uf_rank[rt]) uf_rank[rs]++;
          tree_arcs.push_back(e);
          tree_edges++;
        }
      }
    }

    std::vector<int> tree_adj_deg(_node_num, 0);
    for (int k = 0; k < tree_edges; ++k) {
      ArcsType e = tree_arcs[k];
      tree_adj_deg[arcSource(e)]++;
      tree_adj_deg[arcTarget(e)]++;
    }
    std::vector<int> tree_adj_start(_node_num + 1, 0);
    for (int i = 0; i < _node_num; ++i) {
      tree_adj_start[i + 1] = tree_adj_start[i] + tree_adj_deg[i];
    }
    int total_adj = tree_adj_start[_node_num];
    std::vector<int> tree_adj_node(total_adj);
    std::vector<ArcsType> tree_adj_arc(total_adj);
    std::vector<int> tree_adj_pos(_node_num, 0);
    for (int k = 0; k < tree_edges; ++k) {
      ArcsType e = tree_arcs[k];
      int s = arcSource(e), t = arcTarget(e);
      int ps = tree_adj_start[s] + tree_adj_pos[s]++;
      tree_adj_node[ps] = t;
      tree_adj_arc[ps] = e;
      int pt = tree_adj_start[t] + tree_adj_pos[t]++;
      tree_adj_node[pt] = s;
      tree_adj_arc[pt] = e;
    }

    _search_arc_num = _arc_num;
    _all_arc_num = _arc_num + _node_num;
    _root = _node_num;

    for (ArcsType u = 0, e = _arc_num; u != _node_num; ++u, ++e) {
      setArcState(e, STATE_TREE);
      if (_supply[u] >= 0) {
        setArcEndpoints(e, u, _root);
        setArcCost(e, 0);
        setArcFlow(e, _supply[u]);
      } else {
        setArcEndpoints(e, _root, u);
        setArcCost(e, ART_COST);
        setArcFlow(e, -_supply[u]);
      }
    }

    _parent[_root] = -1;
    _pred[_root] = -1;
    _supply[_root] = -_sum_supply;
    _pi[_root] = 0;

    std::vector<bool> is_rep(_node_num, false);
    std::vector<bool> visited(_node_num, false);

    for (int u = 0; u < _node_num; ++u) {
      if (visited[u]) continue;
      is_rep[u] = true;

      _parent[u] = _root;
      _pred[u] = _arc_num + u;
      _forward[u] = (_supply[u] >= 0);
      setArcState(_arc_num + u, STATE_TREE);
      visited[u] = true;

      std::queue<int> bfs_queue;
      bfs_queue.push(u);
      while (!bfs_queue.empty()) {
        int v = bfs_queue.front();
        bfs_queue.pop();
        for (int k = tree_adj_start[v]; k < tree_adj_start[v + 1]; ++k) {
          int w = tree_adj_node[k];
          ArcsType arc_e = tree_adj_arc[k];
          if (visited[w]) continue;
          visited[w] = true;

          _parent[w] = v;
          _pred[w] = arc_e;
          setArcState(arc_e, STATE_TREE);
          _forward[w] = (arcSource(arc_e) == w);

          setArcState(_arc_num + w, STATE_LOWER);
          setArcFlow(_arc_num + w, 0);

          bfs_queue.push(w);
        }
      }
    }

    {
      std::vector<std::vector<int>> children(_node_num + 1);
      for (int u = 0; u < _node_num; ++u) {
        children[_parent[u]].push_back(u);
      }

      std::vector<int> preorder;
      preorder.reserve(_node_num + 1);
      std::stack<int> dfs_stack;
      dfs_stack.push(_root);
      while (!dfs_stack.empty()) {
        int v = dfs_stack.top();
        dfs_stack.pop();
        preorder.push_back(v);
        for (int i = static_cast<int>(children[v].size()) - 1; i >= 0; --i) {
          dfs_stack.push(children[v][i]);
        }
      }

      for (size_t i = 0; i < preorder.size() - 1; ++i) {
        _thread[preorder[i]] = preorder[i + 1];
      }
      _thread[preorder.back()] = preorder[0];

      for (int u = 0; u <= _node_num; ++u) {
        _rev_thread[_thread[u]] = u;
      }

      for (int u = 0; u <= _node_num; ++u) {
        _succ_num[u] = 1;
      }
      for (int i = static_cast<int>(preorder.size()) - 1; i > 0; --i) {
        int u = preorder[i];
        _succ_num[_parent[u]] += _succ_num[u];
      }

      std::vector<int> pos(_node_num + 1);
      for (size_t i = 0; i < preorder.size(); ++i) {
        pos[preorder[i]] = static_cast<int>(i);
      }
      for (size_t i = 0; i < preorder.size(); ++i) {
        int u = preorder[i];
        _last_succ[u] = preorder[pos[u] + _succ_num[u] - 1];
      }
    }

    {
      std::vector<Value> net(_node_num + 1);
      for (int u = 0; u <= _node_num; ++u) {
        net[u] = _supply[u];
      }

      std::vector<int> preorder;
      preorder.reserve(_node_num + 1);
      int cur = _root;
      for (int i = 0; i <= _node_num; ++i) {
        preorder.push_back(cur);
        cur = _thread[cur];
      }

      int ejected = 0;
      for (int i = static_cast<int>(preorder.size()) - 1; i > 0; --i) {
        int u = preorder[i];
        ArcsType e = _pred[u];

        Value f = _forward[u] ? net[u] : -net[u];

        if (f >= 0) {
          setArcFlow(e, f);
          net[_parent[u]] += net[u];
        } else {
          if (e < _arc_num) {
            setArcState(e, STATE_LOWER);
            setArcFlow(e, 0);
          }
          ArcsType art_e = _arc_num + u;
          _parent[u] = _root;
          _pred[u] = art_e;
          _forward[u] = (arcSource(art_e) == u);
          setArcState(art_e, STATE_TREE);

          Value art_f = _forward[u] ? net[u] : -net[u];
          setArcFlow(art_e, art_f >= 0 ? art_f : -art_f);
          if (art_f < 0) {
            _forward[u] = !_forward[u];
            setArcFlow(art_e, -art_f);
          }

          net[_root] += net[u];
          ejected++;
        }
      }
      if (ejected > 0) {
        std::vector<std::vector<int>> children2(_node_num + 1);
        for (int u = 0; u < _node_num; ++u) {
          children2[_parent[u]].push_back(u);
        }
        std::vector<int> preorder2;
        preorder2.reserve(_node_num + 1);
        std::stack<int> dfs2;
        dfs2.push(_root);
        while (!dfs2.empty()) {
          int v = dfs2.top();
          dfs2.pop();
          preorder2.push_back(v);
          for (int j = static_cast<int>(children2[v].size()) - 1; j >= 0;
               --j) {
            dfs2.push(children2[v][j]);
          }
        }
        for (size_t i = 0; i < preorder2.size() - 1; ++i) {
          _thread[preorder2[i]] = preorder2[i + 1];
        }
        _thread[preorder2.back()] = preorder2[0];
        for (int u = 0; u <= _node_num; ++u) {
          _rev_thread[_thread[u]] = u;
        }
        for (int u = 0; u <= _node_num; ++u) _succ_num[u] = 1;
        for (int i = static_cast<int>(preorder2.size()) - 1; i > 0; --i) {
          _succ_num[_parent[preorder2[i]]] += _succ_num[preorder2[i]];
        }
        std::vector<int> pos2(_node_num + 1);
        for (size_t i = 0; i < preorder2.size(); ++i)
          pos2[preorder2[i]] = static_cast<int>(i);
        for (size_t i = 0; i < preorder2.size(); ++i) {
          int u = preorder2[i];
          _last_succ[u] = preorder2[pos2[u] + _succ_num[u] - 1];
        }
      }
    }

    {
      _pi[_root] = 0;
      int u = _thread[_root];
      while (u != _root) {
        ArcsType e = _pred[u];
        int v = _parent[u];
        Cost c = getCostForArc(e);
        if (_forward[u]) {
          _pi[u] = _pi[v] - c;
        } else {
          _pi[u] = _pi[v] + c;
        }
        u = _thread[u];
      }
    }

    in_arc = 0;
    return true;
  }

  bool init() {
    if (_node_num == 0) return false;

    _sum_supply = 0;
    for (int i = 0; i != _node_num; ++i) {
      _sum_supply += _supply[i];
    }
    if (std::abs(static_cast<double>(_sum_supply)) > _EPSILON) return false;

    _sum_supply = 0;

    Cost ART_COST;
    if (std::numeric_limits<Cost>::is_exact) {
      ART_COST = std::numeric_limits<Cost>::max() / 2 + 1;
    } else {
      Cost max_cost = maxRealArcCost();
      ART_COST = (max_cost + 1) * _node_num;
    }

    fillArcStates(_arc_num, STATE_LOWER);

    _root = _node_num;
    _parent[_root] = -1;
    _pred[_root] = -1;
    _thread[_root] = 0;
    _rev_thread[0] = _root;
    _succ_num[_root] = _node_num + 1;
    _last_succ[_root] = _root - 1;
    _supply[_root] = -_sum_supply;
    _pi[_root] = 0;

    if (_sum_supply == 0) {
      _search_arc_num = _arc_num;
      _all_arc_num = _arc_num + _node_num;
      for (ArcsType u = 0, e = _arc_num; u != _node_num; ++u, ++e) {
        _parent[u] = _root;
        _pred[u] = e;
        _thread[u] = static_cast<int>(u + 1);
        _rev_thread[u + 1] = static_cast<int>(u);
        _succ_num[u] = 1;
        _last_succ[u] = static_cast<int>(u);
        setArcState(e, STATE_TREE);
        if (_supply[u] >= 0) {
          _forward[u] = true;
          _pi[u] = 0;
          setArcEndpoints(e, static_cast<int>(u), _root);
          setArcFlow(e, _supply[u]);
          setArcCost(e, 0);
        } else {
          _forward[u] = false;
          _pi[u] = ART_COST;
          setArcEndpoints(e, _root, static_cast<int>(u));
          setArcFlow(e, -_supply[u]);
          setArcCost(e, ART_COST);
        }
      }
    } else if (_sum_supply > 0) {
      _search_arc_num = _arc_num + _node_num;
      ArcsType f = _arc_num + _node_num;
      for (ArcsType u = 0, e = _arc_num; u != _node_num; ++u, ++e) {
        _parent[u] = _root;
        _thread[u] = static_cast<int>(u + 1);
        _rev_thread[u + 1] = static_cast<int>(u);
        _succ_num[u] = 1;
        _last_succ[u] = static_cast<int>(u);
        if (_supply[u] >= 0) {
          _forward[u] = true;
          _pi[u] = 0;
          _pred[u] = e;
          setArcEndpoints(e, static_cast<int>(u), _root);
          setArcFlow(e, _supply[u]);
          setArcCost(e, 0);
          setArcState(e, STATE_TREE);
        } else {
          _forward[u] = false;
          _pi[u] = ART_COST;
          _pred[u] = f;
          setArcEndpoints(f, _root, static_cast<int>(u));
          setArcFlow(f, -_supply[u]);
          setArcCost(f, ART_COST);
          setArcState(f, STATE_TREE);
          setArcEndpoints(e, static_cast<int>(u), _root);
          setArcCost(e, 0);
          setArcState(e, STATE_LOWER);
          ++f;
        }
      }
      _all_arc_num = f;
    } else {
      _search_arc_num = _arc_num + _node_num;
      ArcsType f = _arc_num + _node_num;
      for (ArcsType u = 0, e = _arc_num; u != _node_num; ++u, ++e) {
        _parent[u] = _root;
        _thread[u] = static_cast<int>(u + 1);
        _rev_thread[u + 1] = static_cast<int>(u);
        _succ_num[u] = 1;
        _last_succ[u] = static_cast<int>(u);
        if (_supply[u] <= 0) {
          _forward[u] = false;
          _pi[u] = 0;
          _pred[u] = e;
          setArcEndpoints(e, _root, static_cast<int>(u));
          setArcFlow(e, -_supply[u]);
          setArcCost(e, 0);
          setArcState(e, STATE_TREE);
        } else {
          _forward[u] = true;
          _pi[u] = -ART_COST;
          _pred[u] = f;
          setArcEndpoints(f, static_cast<int>(u), _root);
          setArcFlow(f, _supply[u]);
          setArcState(f, STATE_TREE);
          setArcCost(f, ART_COST);
          setArcEndpoints(e, _root, static_cast<int>(u));
          setArcCost(e, 0);
          setArcState(e, STATE_LOWER);
          ++f;
        }
      }
      _all_arc_num = f;
    }

    return true;
  }

  void findJoinNode() {
    int u = arcSource(in_arc);
    int v = arcTarget(in_arc);
    while (u != v) {
      if (_succ_num[u] < _succ_num[v]) {
        u = _parent[u];
      } else {
        v = _parent[v];
      }
    }
    join = static_cast<ArcsType>(u);
  }

  bool findLeavingArc() {
    if (arcState(in_arc) == STATE_LOWER) {
      first = static_cast<ArcsType>(arcSource(in_arc));
      second = static_cast<ArcsType>(arcTarget(in_arc));
    } else {
      first = static_cast<ArcsType>(arcTarget(in_arc));
      second = static_cast<ArcsType>(arcSource(in_arc));
    }
    delta = INF;
    char result = 0;
    Value d;
    ArcsType e;

    for (int u = static_cast<int>(first); u != static_cast<int>(join);
         u = _parent[u]) {
      e = _pred[u];
      d = _forward[u] ? arcFlow(e) : INF;
      if (d < delta) {
        delta = d;
        u_out = static_cast<ArcsType>(u);
        result = 1;
      }
    }
    for (int u = static_cast<int>(second); u != static_cast<int>(join);
         u = _parent[u]) {
      e = _pred[u];
      d = _forward[u] ? INF : arcFlow(e);
      if (d <= delta) {
        delta = d;
        u_out = static_cast<ArcsType>(u);
        result = 2;
      }
    }

    if (result == 1) {
      u_in = first;
      v_in = second;
    } else {
      u_in = second;
      v_in = first;
    }
    return result != 0;
  }

  void changeFlow(bool change) {
    if (delta > 0) {
      Value val = arcState(in_arc) * delta;
      addArcFlow(in_arc, val);
      for (int u = arcSource(in_arc); u != static_cast<int>(join);
           u = _parent[u]) {
        addArcFlow(_pred[u], _forward[u] ? -val : val);
      }
      for (int u = arcTarget(in_arc); u != static_cast<int>(join);
           u = _parent[u]) {
        addArcFlow(_pred[u], _forward[u] ? val : -val);
      }
    }
    if (change) {
      setArcState(in_arc, STATE_TREE);
      setArcState(_pred[u_out],
                  (arcFlow(_pred[u_out]) == 0) ? STATE_LOWER : STATE_UPPER);
    } else {
      flipArcState(in_arc);
    }
  }

  void updateTreeStructure() {
    int u = 0;
    int w = 0;
    int old_rev_thread = _rev_thread[u_out];
    int old_succ_num = _succ_num[u_out];
    int old_last_succ = _last_succ[u_out];
    v_out = static_cast<ArcsType>(_parent[u_out]);

    u = _last_succ[u_in];
    right = static_cast<ArcsType>(_thread[static_cast<std::size_t>(u)]);

    if (old_rev_thread == static_cast<int>(v_in)) {
      last = static_cast<ArcsType>(_thread[_last_succ[u_out]]);
    } else {
      last = static_cast<ArcsType>(_thread[v_in]);
    }

    _thread[v_in] = static_cast<int>(stem = u_in);
    _dirty_revs.clear();
    _dirty_revs.push_back(static_cast<int>(v_in));
    par_stem = v_in;
    while (stem != u_out) {
      new_stem = static_cast<ArcsType>(_parent[stem]);
      _thread[u] = static_cast<int>(new_stem);
      _dirty_revs.push_back(u);

      w = _rev_thread[stem];
      _thread[w] = static_cast<int>(right);
      _rev_thread[right] = w;

      _parent[stem] = static_cast<int>(par_stem);
      par_stem = stem;
      stem = new_stem;

      u = _last_succ[stem] == _last_succ[par_stem] ? _rev_thread[par_stem]
                                                   : _last_succ[stem];
      right = static_cast<ArcsType>(_thread[u]);
    }
    _parent[u_out] = static_cast<int>(par_stem);
    _thread[u] = static_cast<int>(last);
    _rev_thread[last] = u;
    _last_succ[u_out] = u;

    if (old_rev_thread != static_cast<int>(v_in)) {
      _thread[old_rev_thread] = static_cast<int>(right);
      _rev_thread[right] = old_rev_thread;
    }

    for (std::size_t i = 0; i != _dirty_revs.size(); ++i) {
      int u_idx = _dirty_revs[i];
      _rev_thread[_thread[u_idx]] = u_idx;
    }

    int tmp_sc = 0;
    int tmp_ls = _last_succ[u_out];
    u = static_cast<int>(u_out);
    while (u != static_cast<int>(u_in)) {
      w = _parent[u];
      _pred[u] = _pred[w];
      _forward[u] = !_forward[w];
      tmp_sc += _succ_num[u] - _succ_num[w];
      _succ_num[u] = tmp_sc;
      _last_succ[w] = tmp_ls;
      u = w;
    }
    _pred[u_in] = in_arc;
    _forward[u_in] = (u_in == static_cast<ArcsType>(arcSource(in_arc)));
    _succ_num[u_in] = old_succ_num;

    int up_limit_in = -1;
    int up_limit_out = -1;
    if (_last_succ[join] == static_cast<int>(v_in)) {
      up_limit_out = static_cast<int>(join);
    } else {
      up_limit_in = static_cast<int>(join);
    }

    for (u = static_cast<int>(v_in);
         u != up_limit_in && _last_succ[u] == static_cast<int>(v_in);
         u = _parent[u]) {
      _last_succ[u] = _last_succ[u_out];
    }
    if (join != static_cast<ArcsType>(old_rev_thread) &&
        v_in != static_cast<ArcsType>(old_rev_thread)) {
      for (u = static_cast<int>(v_out);
           u != up_limit_out && _last_succ[u] == old_last_succ;
           u = _parent[u]) {
        _last_succ[u] = old_rev_thread;
      }
    } else {
      for (u = static_cast<int>(v_out);
           u != up_limit_out && _last_succ[u] == old_last_succ;
           u = _parent[u]) {
        _last_succ[u] = _last_succ[u_out];
      }
    }

    for (u = static_cast<int>(v_in); u != static_cast<int>(join);
         u = _parent[u]) {
      _succ_num[u] += old_succ_num;
    }
    for (u = static_cast<int>(v_out); u != static_cast<int>(join);
         u = _parent[u]) {
      _succ_num[u] -= old_succ_num;
    }
  }

  void updatePotential() {
    Cost sigma = _forward[u_in]
                     ? _pi[v_in] - _pi[u_in] - getCostForArc(_pred[u_in])
                     : _pi[v_in] - _pi[u_in] + getCostForArc(_pred[u_in]);
    int end = _thread[_last_succ[u_in]];
    for (int u = static_cast<int>(u_in); u != end; u = _thread[u]) {
      _pi[u] += sigma;
    }
  }

  bool initialPivots() {
    Value curr, total = 0;
    std::vector<Node> supply_nodes, demand_nodes;
    Node u;
    _graph.first(u);
    for (; u != INVALIDNODE; _graph.next(u)) {
      curr = _supply[_node_id(u)];
      if (curr > 0) {
        total += curr;
        supply_nodes.push_back(u);
      } else if (curr < 0) {
        demand_nodes.push_back(u);
      }
    }
    if (_sum_supply > 0) total -= _sum_supply;
    if (total <= 0) return true;

    ArcVector arc_vector;
    if (_sum_supply >= 0) {
      if (supply_nodes.size() == 1 && demand_nodes.size() == 1) {
        BoolVector reached(_node_num, false);
        Node s = supply_nodes[0], t = demand_nodes[0];
        std::vector<Node> stack;
        reached[t] = true;
        stack.push_back(t);
        while (!stack.empty()) {
          Node u_curr, v_curr = stack.back();
          stack.pop_back();
          if (v_curr == s) break;
          Arc a;
          _graph.firstIn(a, v_curr);
          for (; a != INVALID; _graph.nextIn(a)) {
            u_curr = _graph.source(a);
            if (u_curr < 0 || u_curr >= static_cast<Node>(_node_num)) continue;
            if (reached[static_cast<std::size_t>(u_curr)]) continue;
            ArcsType j = getArcID(a);
            if (INF >= total) {
              arc_vector.push_back(j);
              reached[static_cast<std::size_t>(u_curr)] = 1;
              stack.push_back(u_curr);
            }
          }
        }
      } else {
        for (size_t i = 0; i != demand_nodes.size(); ++i) {
          Node v_curr = demand_nodes[i];
          Cost c, min_cost = std::numeric_limits<Cost>::max();
          Arc min_arc = INVALID;
          Arc a;
          _graph.firstIn(a, v_curr);
          for (; a != INVALID; _graph.nextIn(a)) {
            c = getCostForArc(getArcID(a));
            if (c < min_cost) {
              min_cost = c;
              min_arc = a;
            }
          }
          if (min_arc != INVALID) {
            arc_vector.push_back(getArcID(min_arc));
          }
        }
      }
    } else {
      for (size_t i = 0; i != supply_nodes.size(); ++i) {
        Node u_curr = supply_nodes[i];
        Cost c, min_cost = std::numeric_limits<Cost>::max();
        Arc min_arc = INVALID;
        Arc a;
        _graph.firstOut(a, u_curr);
        for (; a != INVALID; _graph.nextOut(a)) {
          c = getCostForArc(getArcID(a));
          if (c < min_cost) {
            min_cost = c;
            min_arc = a;
          }
        }
        if (min_arc != INVALID) {
          arc_vector.push_back(getArcID(min_arc));
        }
      }
    }

    for (size_t i = 0; i != arc_vector.size(); ++i) {
      in_arc = arc_vector[i];
      if (arcState(in_arc) * (getCostForArc(in_arc) + _pi[arcSource(in_arc)] -
                              _pi[arcTarget(in_arc)]) >=
          0)
        continue;
      findJoinNode();
      bool change = findLeavingArc();
      if (delta >= MAX) return false;
      changeFlow(change);
      if (change) {
        updateTreeStructure();
        updatePotential();
      }
    }
    return true;
  }

  ProblemType start() { return start<BlockSearchPivotRule>(); }

  template <typename PivotRuleImpl>
  ProblemType start() {
    PivotRuleImpl pivot(*this);
    ProblemType retVal = OPTIMAL;

    if (!_warmstart_tree_built) {
      if (!initialPivots()) {
        return UNBOUNDED;
      }
    }

    uint64_t iter_number = 0;
    while (pivot.findEnteringArc()) {
      if (max_iter > 0 && ++iter_number >= max_iter) {
        retVal = MAX_ITER_REACHED;
        break;
      }

      findJoinNode();
      bool change = findLeavingArc();
      if (delta >= MAX) {
        return UNBOUNDED;
      }
      changeFlow(change);
      if (change) {
        updateTreeStructure();
        updatePotential();
      }
    }

    if (retVal == OPTIMAL) {
      for (ArcsType e = _search_arc_num; e != _all_arc_num; ++e) {
        if (arcFlow(e) != 0) {
          if (std::abs(static_cast<double>(arcFlow(e))) > _EPSILON) {
            return INFEASIBLE;
          }
          setArcFlow(e, 0);
        }
      }
    }

    if (_sum_supply == 0) {
      if (_stype == GEQ) {
        Cost max_pot = -std::numeric_limits<Cost>::max();
        for (ArcsType i = 0; i != _node_num; ++i) {
          if (_pi[i] > max_pot) {
            max_pot = _pi[i];
          }
        }
        if (max_pot > 0) {
          for (ArcsType i = 0; i != _node_num; ++i) {
            _pi[i] -= max_pot;
          }
        }
      } else {
        Cost min_pot = std::numeric_limits<Cost>::max();
        for (ArcsType i = 0; i != _node_num; ++i) {
          if (_pi[i] < min_pot) {
            min_pot = _pi[i];
          }
        }
        if (min_pot < 0) {
          for (ArcsType i = 0; i != _node_num; ++i) {
            _pi[i] -= min_pot;
          }
        }
      }
    }

    return retVal;
  }
};

}  // namespace potlemon

#pragma pop_macro("INVALIDNODE")
#pragma pop_macro("INVALID")
#pragma pop_macro("MIN")
#pragma pop_macro("MAX")
