#include <array>
#include <cmath>
#include <optional>
#include <iostream>

#include "adaptive.h"
#include "cache.h"

adaptive::adaptive(CACHE* cache_ptr)
: champsim::modules::prefetcher(cache_ptr),
  cache(cache_ptr),
  q_table_lru(Q_TABLE_SETS, Q_TABLE_SET_ENTRIES),
  table(TABLE_SETS, TABLE_SET_ENTRIES)
{
    std::cout << "[adaptive] Constructed with:\n";
    std::cout << "  Q_TABLE_SETS = " << Q_TABLE_SETS << ", Q_TABLE_SET_ENTRIES = " << Q_TABLE_SET_ENTRIES << "\n";
    std::cout << "  TABLE_SETS   = " << TABLE_SETS   << ", TABLE_SET_ENTRIES   = " << TABLE_SET_ENTRIES << "\n";
}

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

int adaptive::select_action(const std::array<float, MAX_ACTIONS>& q_values, float epsilon) {
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
                              const std::array<float, MAX_ACTIONS>& curr_q_values, float alpha, float gamma) {
    
    auto q_entry = q_table_lru.check_hit(prev_entry);
    std::array<float, MAX_ACTIONS> q_values;

    if (q_entry.has_value())
        q_values = q_entry->q_values;
    else{
        q_values.fill(0.01f);
        q_values[no_prefetch] = 0.0f;
    }
    float max_next_q = *std::max_element(curr_q_values.begin(), curr_q_values.end());

    q_values[prev_action] += alpha * (reward + gamma * max_next_q - q_values[prev_action]); //Q(s, a) ← Q(s, a) + α * (reward + γ * max(Q(s’, a’)) - Q(s, a))
    // Save updated entry
    QTableEntry updated = prev_entry;
    updated.q_values = q_values;
    q_table_lru.fill(updated);
}

bool adaptive::issue_stride_prefetch(champsim::address addr, int stride, int degree) { 
    champsim::address pf_addr = addr;
    if (stride == 0) return true; 
    bool success = false;
    bool light_load = intern_->get_mshr_occupancy_ratio() < 0.5;
    for (int i = 0; i < degree; ++i) {
        pf_addr += stride;
        if (champsim::page_number{pf_addr} != champsim::page_number{addr})
            break;
        success |= prefetch_line(pf_addr, light_load, 0);
        if (success) std::cerr << "Issued stride pf at addr=" << pf_addr << " stride=" << stride << "\n";
    }
    return success;
}


bool adaptive::issue_multi_stride_prefetch(champsim::address ip, champsim::address addr, int degree) {
    auto it = table.check_hit(RLState{ip});
    if (!it.has_value()) return false;
    const RLState& state = it.value();
    champsim::block_number base_line{addr};
    bool success = false;

    for (std::size_t i = 0; i + degree < state.stride_history.size(); ++i) {
        if (state.stride_history[i] == state.last_stride) {
            champsim::block_number block_addr = base_line;

            for (std::size_t j = 1; j <= degree; ++j) {
                auto next_stride = state.stride_history[i + j];
                block_addr += next_stride;

                champsim::address pf_addr{block_addr};
                if (champsim::page_number{pf_addr} != champsim::page_number{addr})
                    break;

                bool light_load = intern_->get_mshr_occupancy_ratio() < 0.5;
                success |= prefetch_line(pf_addr, light_load, 0);
                if (success) std::cerr << "Issued multi stride pf at addr=" << pf_addr << "\n";
            }
            break;
        }
    }
    return success;
}

bool adaptive::issue_locality_prefetch(champsim::address addr, int degree) {
    champsim::block_number cl_addr{addr};
    bool light_load = intern_->get_mshr_occupancy_ratio() < 0.5;
    bool success = false;
    for (int i = 1; i <= degree; ++i) {
        champsim::address pf_addr{cl_addr + i};
        if (champsim::page_number{pf_addr} != champsim::page_number{addr})
            break;
        success |= prefetch_line(pf_addr, light_load, 0);
        if (success) std::cerr << "Issued locality pf at addr=" << pf_addr << "\n";
    }
    for (int i = 1; i <= degree; ++i) {
        champsim::address pf_addr{cl_addr - i};
        if (champsim::page_number{pf_addr} != champsim::page_number{addr})
            break;
        success |= prefetch_line(pf_addr, light_load, 0);
        if (success) std::cerr << "Issued locality pf at addr=" << pf_addr << "\n";
    }
    return success;
}

bool adaptive::issue_correlation_prefetch(champsim::address ip, champsim::address addr) {
    auto it = table.check_hit(RLState{ip});
    bool light_load = intern_->get_mshr_occupancy_ratio() < 0.5;
    bool success = false;
    if (!it.has_value()) return false;
    const RLState& state = it.value();
    if (state.recent_addr.size() < 2)
        return true;
    for (size_t i = 0; i + 1 < state.recent_addr.size(); ++i) {
        if (state.recent_addr[i] == champsim::block_number{addr}) {
            champsim::block_number next = state.recent_addr[i + 1];
            if (champsim::page_number{next} == champsim::page_number{addr}) {
                success |= prefetch_line(champsim::address{next}, light_load, 0);
                if (success) std::cerr << "Issued correlation pf at addr=" << next << "\n";
            }
            break;
        }
    }
    return success;
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

  //std::cerr << "Entered prefetcher_cache_operate" << std::endl;

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
    std::rotate(state.recent_addr.begin(), state.recent_addr.begin() + 1, state.recent_addr.end());
    state.recent_addr.back() = cl_addr;
  }
  else{
    state.ip = ip;
    state.last_cl_addr = cl_addr;
    state.last_stride = stride;
    state.stride_history.back() = stride;
    state.recent_addr.back() = cl_addr;
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
  PrefetchTask task{ip, addr, action, stride};
  prefetch_queue.push_back(task);
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

uint32_t adaptive::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in){
/*
| Parameter      | Meaning                                                               |
| -------------- | --------------------------------------------------------------------- |
| `addr`         | The **address** be+where this line is inserted                         |
| `way`          | The **way** inside the set (for set-associative caches)               |
| `prefetch`     | Is this line a **prefetch** (1) or a **demand load** (0)?             |
| `evicted_addr` | If something was **evicted** to make space, this is its address       |
| `metadata_in`  | Metadata returned from `cache_operate` — can be used for tracking     |
*/
  return metadata_in;
}

void adaptive::prefetcher_cycle_operate(){

    //std::cerr << "Entered prefetcher_cycle_operate" << std::endl;
    
    int issued = 0;
    while (!prefetch_queue.empty() && issued < 1) {
        PrefetchTask& task = prefetch_queue.front(); 
        bool success = false;
        switch (static_cast<PrefetchActions>(task.action)) {
            case no_prefetch:
                success = true; // no work to do
                break;
            case simple_stride:
                success = issue_stride_prefetch(task.addr, task.stride);
                break;
            case multi_stride:
                success = issue_multi_stride_prefetch(task.ip, task.addr);
                break;
            case locality:
                success = issue_locality_prefetch(task.addr);
                break;
            case correlation:
                success = issue_correlation_prefetch(task.ip,task.addr);
                break;
            default:
                break;
        }
        if (success) {
            prefetch_queue.pop_front();  // remove once prefetch was handled
            ++issued;
        } else {
            break; // MSHR is full or something failed, try again next cycle
        }
    }
  //Only useful if you want to decay confidence, update timers, or do background scanning every cycle. 
}
