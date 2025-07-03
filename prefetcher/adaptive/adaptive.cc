#include <array>
#include <cmath>
#include <optional>

#include "adaptive.h"
#include "cache.h"

void update_confidence(float& confidence, bool detected, bool useful) {
    if (detected && confidence < 0.5f)
        confidence = 0.5f;
    if (useful)
        confidence = std::min(1.0f, confidence + 0.1f);
    else
        confidence *= 0.9f;
}

// Returns tile indices for a feature value
std::vector<int> tile_indices(int feature_value, int num_tilings, int tiles_per_tiling) {
    std::vector<int> indices;
    for (int tiling = 0; tiling < num_tilings; ++tiling) {
        int offset = tiling * 97; // prime offset to reduce collisions
        int index = ((feature_value + offset) / tiles_per_tiling) + tiling * 1000;
        indices.push_back(index);
    }
    return indices;
}

std::vector<int> adaptive::RLState::active_tiles() const {
    std::vector<int> indices;

    // 1. IP Encoding
    uint64_t ip_val = ip.to<uint64_t>() >> 6;
    auto ip_tiles = tile_indices(ip_val % 1024, 4, 32);
    indices.insert(indices.end(), ip_tiles.begin(), ip_tiles.end());

    // 2. Stride History Encoding
    for (const auto& stride : stride_history) {
        auto stride_tiles = tile_indices(static_cast<int>(stride), 3, 16);
        indices.insert(indices.end(), stride_tiles.begin(), stride_tiles.end());
    }

    return indices;
}

int select_action(const std::array<float, MAX_ACTIONS>& q_values, float epsilon) {
    float random_val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);  // Generate a random float value
    if (random_val < epsilon) {
      return rand() % MAX_ACTIONS; // Explore the actions 
    }
    else{
        return std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end())); // Exploit the action with max Q_value
    }
}

float adaptive::compute_reward(access_type type, uint8_t cache_hit, bool useful_prefetch) {
    if (type == access_type::LOAD || type == access_type::RFO) {
        if (cache_hit && useful_prefetch)
            return +1.0f; // useful prefetch
        else if (!cache_hit)
            return -1.0f; // demand miss not covered
        else
            return 0.0f;  // load hit, but not due to prefetch
    }
    return 0.0f; // for WRITE or PREFETCH accesses
}

void adaptive::update_q_value(const QTableEntry& prev_entry, int prev_action, float reward, const QTableEntry& curr_entry,
                              const std::array<float, MAX_ACTIONS>& curr_q_values, float alpha = 0.1f, float gamma = 0.9f) {
    
    auto q_entry = q_table_lru.check_hit(prev_entry);
    std::array<float, MAX_ACTIONS> q_values;

    if (q_entry.has_value())
        q_values = q_entry->q_values;
    else
        q_values.fill(0.0f);

    float max_next_q = *std::max_element(curr_q_values.begin(), curr_q_values.end());

    q_values[prev_action] += alpha * (reward + gamma * max_next_q - q_values[prev_action]); //Q(s, a) ← Q(s, a) + α * (reward + γ * max(Q(s’, a’)) - Q(s, a))
    // Save updated entry
    QTableEntry updated = prev_entry;
    updated.q_values = q_values;
    q_table_lru.fill(updated);
}

void adaptive::issue_stride_prefetch(champsim::address addr, int stride, int degree = 2) { 
    champsim::address pf_address = addr;
    bool light_load = intern_->get_mshr_occupancy_ratio() < 0.5; // miss status holding registers (MSHR) 
    for (int i = 0; i < degree; ++i) {                           //Issue stride prefetch
        pf_address += stride;
        if (champsim::page_number{pf_address} != champsim::page_number{addr})
            break; // avoid crossing page
        prefetch_line(pf_address, light_load, 0);
    }
}


uint32_t adaptive::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type, uint32_t metadata_in){
  // SO we need to observe the pattern of access through this function
/*
| Parameter         | Meaning                                                              |
| ----------------- | -------------------------------------------------------------------- |
| `addr`            | The memory **address being accessed** right no+w                     |
| `ip`              | The **instruction pointer (IP)** of the instruction doing the access |
| `cache_hit`       | Whether this access was a **cache hit** (non-zero) or **miss** (0)   |
| `useful_prefetch` | Was this access satisfied by a **prefetch that arrived earlier**?    |
| `type`            | Type of access: **demand**, **prefetch**, **writeback**, etc.        |
| `metadata_in`     | Info passed along from previous events (usually just passed through) |
*/ 
  champsim::block_number cl_addr{addr};
  champsim::block_number::difference_type stride = 0;
  RLState state;
  static QTableEntry prev_entry;
  static int prev_action = -1;
  
  auto entry = table.check_hit(RLState{ip}); // first we insert the RLState struct in the lru. we only need IP to find matching entry
  if (entry.has_value()) {
    state = entry.value();
    // update stride history
    stride = offset(state.last_cl_addr, cl_addr); // compare with the last stride and check for simple stride pattern
    state.last_stride = stride;
    state.last_cl_addr = cl_addr;
    std::rotate(state.stride_history.begin(), state.stride_history.begin() + 1, state.stride_history.end());
    state.stride_history.back() = stride;
  }
  else{
    state.ip = ip;
    state.last_cl_addr = cl_addr;
    state.last_stride = stride;
    state.stride_history.back() = stride;
  }
  table.fill(state);

  std::vector<int> active_indices = state.active_tiles();   //Quantize input to state vector
  QTableEntry q_entry{.key = active_indices};               // Hash struct for lru table
  std::array<float, MAX_ACTIONS> q_values{};
  auto q_table_hit = q_table_lru.check_hit(q_entry);        // check Q_table entries
  if (q_table_hit.has_value()) {
      q_values = q_table_hit->q_values;
  } else {
      q_values.fill(0.0f); // initialize unseen state
  }

  int action = select_action(q_values, 0.2f);  // Take a action 20% explore

  switch (static_cast<PrefetchActions>(action)) {
    case no_prefetch:// Do nothing
      break;
    case simple_stride: issue_stride_prefetch(addr, stride); 
      break;
    case multi_stride: issue_multi_stride_prefetch(addr);
      break;
    case locality: issue_locality_prefetch(addr);
      break;
    case correlation: issue_correlation_prefetch(addr);
      break;
    default:// fallback
      break;
  }
  // assign reward and update the q table 
  if (prev_action != -1) {
      float reward = compute_reward(type, cache_hit, useful_prefetch);
      update_q_value(prev_entry, prev_action, reward, q_entry, q_values);
  }

  // Now save for next update
  prev_entry = q_entry;
  prev_action = action;
  state.last_action = action;
  return metadata_in;
}

uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in){
/*
| Parameter      | Meaning                                                               |
| -------------- | --------------------------------------------------------------------- |
| `addr`         | The **address** be+where this line is inserted                         |
| `way`          | The **way** inside the set (for set-associative caches)               |
| `prefetch`     | Is this line a **prefetch** (1) or a **demand load** (0)?             |
| `evicted_addr` | If something was **evicted** to make space, this is its address       |
| `metadata_in`  | Metadata returned from `cache_operate` — can be used for tracking     |
*/
}

void prefetcher_cycle_operate(){
  //Only useful if you want to decay confidence, update timers, or do background scanning every cycle. 
}
