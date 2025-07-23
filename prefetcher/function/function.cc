#include <array>
#include <cmath>
#include <optional>
#include <iostream>

#include "function.h"
#include "cache.h"

function::function(CACHE* cache_ptr)
: champsim::modules::prefetcher(cache_ptr),
  cache(cache_ptr),
  table(TABLE_SETS, TABLE_SET_ENTRIES)
{
    std::cout << "[function] Constructed with:\n";
    std::cout << "  TABLE_SETS   = " << TABLE_SETS   << ", TABLE_SET_ENTRIES   = " << TABLE_SET_ENTRIES << "\n";
    for (auto& row : q_table) {
        row.fill(0.1f);
        row[no_prefetch] = 0.0f;
    }
}
function::RecentPrefetchHistory pf_history;
function::PrefetchStats stats;

constexpr int function::stride_to_bin(champsim::block_number::difference_type stride)
{
    if (stride < STRIDE_MIN)  return 0;
    if (stride > STRIDE_MAX)  return NUM_STRIDE_BINS - 1;
    return static_cast<int>(stride) - STRIDE_MIN;   // shift into 0…NUM_BINS‑1
}

constexpr int function::encode_state(champsim::block_number::difference_type stride, const std::array<champsim::block_number::difference_type, max_history>& stride_history)
{
    int idx = 0;
    idx = idx * NUM_STRIDE_BINS + stride_to_bin(stride);
    for (int i = 0; i < max_history; ++i) {
        idx = idx * NUM_STRIDE_BINS + stride_to_bin(stride_history[i]);
    }
    return idx;
}

int function::select_action(const std::array<float, MAX_ACTIONS>& q_values, float epsilon) {
    float random_val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);  // Generate a random float value
    if (random_val < epsilon) {
      return rand() % MAX_ACTIONS; // Explore the actions 
    }
    else{
        return std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end())); // Exploit the action with max Q_value
    }
}

float function::compute_reward(access_type type, uint8_t cache_hit, bool useful_prefetch) {
    float reward = 0.0f;
    if (type == access_type::LOAD || type == access_type::RFO) {
        if (cache_hit && useful_prefetch){
            reward = +1.0f; // useful prefetch
            //std::cerr << "Reward: " << reward << " (hit=" << (int)cache_hit << ", useful=" << useful_prefetch << ")\n";

        }
        else if (!cache_hit) {
            reward = -1.0f; // demand miss not covered
            //std::cerr << "Reward: " << reward << " (hit=" << (int)cache_hit << ", useful=" << useful_prefetch << ")\n";
        }
        else
            reward = -0.1f;  // load hit, but not due to prefetch
    }
    return reward; // for WRITE or PREFETCH accesses
}

void function::update_q_value(int prev_index, int prev_action, float reward, int curr_index, float alpha, float gamma)
{
    auto& prev_q = q_table[prev_index];
    const auto& curr_q = q_table[curr_index];

    float max_next_q = *std::max_element(curr_q.begin(), curr_q.end());
    prev_q[prev_action] += alpha * (reward + gamma * max_next_q - prev_q[prev_action]);
}

bool function::issue_stride_prefetch(champsim::address addr, int stride, int degree) { 
    champsim::address pf_addr = addr;
    if (stride == 0) return true; 
    bool success = false;
    bool light_load = intern_->get_mshr_occupancy_ratio() < 0.5;
    for (int i = 0; i < degree; ++i) {
        pf_addr += stride;
        if (champsim::page_number{pf_addr} != champsim::page_number{addr})
            break;
        if (!pf_history.contains(champsim::block_number{pf_addr})){ 
            success |= prefetch_line(pf_addr, light_load, 0);
            pf_history.insert(champsim::block_number{pf_addr});
            stats.stride_calls++;
            
        }
        else return true;
        //if (success) std::cerr << "Issued stride pf at addr=" << pf_addr << " stride=" << stride << "\n";
    }
    return success;
}


bool function::issue_multi_stride_prefetch(champsim::address ip, champsim::address addr, int degree) {
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
                bool light_load = intern_->get_mshr_occupancy_ratio() < 0.5;

                champsim::address pf_addr{block_addr};
                if (champsim::page_number{pf_addr} != champsim::page_number{addr})
                    break;
                if (!pf_history.contains(block_addr)){ 
                    success |= prefetch_line(pf_addr, light_load, 0);
                    pf_history.insert(block_addr);
                    stats.m_stride_calls ++;
                }
                else return true;
                //if (success) std::cerr << "Issued multi stride pf at addr=" << pf_addr << "\n";
            }
            break;
        }
    }
    return success;
}

bool function::issue_locality_prefetch(champsim::address addr, int degree) {
    champsim::block_number block_addr{addr};
    bool light_load = intern_->get_mshr_occupancy_ratio() < 0.5;
    bool success = false;
    for (int i = 1; i <= degree; ++i) {
        champsim::address pf_addr{block_addr + i};
        if (champsim::page_number{pf_addr} != champsim::page_number{addr})
            break;
        if (!pf_history.contains(block_addr + i)){ 
            success |= prefetch_line(pf_addr, light_load, 0);
            pf_history.insert(block_addr + i);
            stats.locality_calls ++;
        }
        else return true;
        //if (success) std::cerr << "Issued locality pf at addr=" << pf_addr << "\n";
    }
    for (int i = 1; i <= degree; ++i) {
        champsim::address pf_addr{block_addr - i};
        if (champsim::page_number{pf_addr} != champsim::page_number{addr})
            break;
        if (!pf_history.contains(block_addr - i)){ 
            success |= prefetch_line(pf_addr, light_load, 0);
            pf_history.insert(block_addr - i);
        }
        else return true;
        //if (success) std::cerr << "Issued locality pf at addr=" << pf_addr << "\n";
    }
    return success;
}

bool function::issue_correlation_prefetch(champsim::address ip, champsim::address addr) {
    auto it = table.check_hit(RLState{ip});
    bool light_load = intern_->get_mshr_occupancy_ratio() < 0.5;
    bool success = false;
    if (!it.has_value()) return false;
    const RLState& state = it.value();
    if (state.recent_addr.size() < 2)
        return true;
    for (size_t i = 0; i + 1 < state.recent_addr.size(); ++i) {
        if (state.recent_addr[i] == champsim::block_number{addr}) {
            champsim::block_number block_addr = state.recent_addr[i + 1];
            if (champsim::page_number{block_addr} != champsim::page_number{addr})
                break;
            if (!pf_history.contains(block_addr)){ 
                success |= prefetch_line(champsim::address{block_addr}, light_load, 0);
                pf_history.insert(block_addr);
                stats.corellation_calls ++;
            }
            else return true;
            //if (success) std::cerr << "Issued correlation pf at addr=" << next << "\n";
        }
    }
    return success;
}
uint32_t function::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type, uint32_t metadata_in){
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
  
  auto entry = table.check_hit(RLState{ip}); // first we insert the RLState struct in the lru. we only need IP to find matching entry
  if (entry.has_value()) {
    state = entry.value();
    // update stride history
    stride = offset(state.last_cl_addr, cl_addr); // compare with the last stride and check for simple stride pattern
    state.last_stride = stride;
    state.last_cl_addr = cl_addr;
    std::rotate(state.stride_history.begin(), state.stride_history.begin() + 1, state.stride_history.end());
    state.stride_history.back() = stride;
    state.stride_count ++;
    if (state.stride_count == max_history) state.stride_history_valid = true;
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

  int q_index = encode_state(state.last_stride, state.stride_history);
  std::array<float, MAX_ACTIONS>& q_values = q_table[q_index];
  int action = 0;
  if (state.stride_history_valid) action = select_action(q_values, 0.2f);  // Take a action 20% explore
  PrefetchTask task{ip, addr, action, stride};
  prefetch_queue.push_back(task);
  // assign reward and update the q table 
  if (state.last_action != -1) {
      float reward = compute_reward(type, cache_hit, useful_prefetch);
      update_q_value(state.last_index, state.last_action, reward, q_index);
  }

  // Now save for next update
  state.last_index = q_index; 
  state.last_action = action;
  
  table.fill(state);
  return metadata_in;
}

uint32_t function::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in){
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
// static int tick = 0;
void function::prefetcher_cycle_operate() {
    int issued = 0;

    while (!prefetch_queue.empty() && issued < 1) {
        //std::cerr << "[cycle] Issued=" << issued << ", MSHR=" << intern_->get_mshr_occupancy_ratio() << ", Queue=" << prefetch_queue.size() << "\n";
        PrefetchTask& task = prefetch_queue.front();
        bool issue_ok = false;
        switch (static_cast<PrefetchActions>(task.action)) {
            case no_prefetch:  issue_ok = true; break;
            case simple_stride:  issue_ok = issue_stride_prefetch(task.addr, task.stride); break;
            case multi_stride:   issue_ok = issue_multi_stride_prefetch(task.ip, task.addr); break;
            case locality:       issue_ok = issue_locality_prefetch(task.addr); break;
            case correlation:    issue_ok = issue_correlation_prefetch(task.ip, task.addr); break;
        }
        prefetch_queue.pop_front();    // Always remove task
        if (issue_ok)
            ++issued;
    }
}

void function::prefetcher_final_stats(){
    std::cout << "Stride Prefetches Issued:" << stats.stride_calls << "\n";
    std::cout << "Multi Stride Prefetches Issued:" << stats.m_stride_calls << "\n";
    std::cout << "Locality Prefetches Issued:" << stats.locality_calls << "\n";
    std::cout << "Corellation Prefetches Issued:" << stats.corellation_calls << "\n";
}