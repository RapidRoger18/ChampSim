#ifndef ADAPTIVE_H
#define ADAPTIVE_H

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <vector>
#include "address.h"
#include "champsim.h"
#include "modules.h"
#include "msl/lru_table.h"

constexpr int MAX_ACTIONS = 4;

struct function : public champsim::modules::prefetcher {

    explicit function(CACHE* cache);
    // Prefetcher API Hooks
    uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type, uint32_t metadata_in);
    uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);
    void prefetcher_cycle_operate();
    void prefetcher_final_stats();

    constexpr static std::size_t TABLE_SETS = 256;   // There are 256 sets (buckets).
    constexpr static std::size_t TABLE_SET_ENTRIES = 4; // Each set can store up to 4 entries.
    // When a 5th IP maps to the same set, one of the 4 entries is evicted (typically Least Recently Used — LRU).
    static constexpr size_t max_history = 4;
    static constexpr size_t max_recent_addr = 8;
struct RLState {

    champsim::address ip{};
    champsim::block_number last_cl_addr{};
    champsim::block_number::difference_type last_stride{};
    std::array<champsim::block_number::difference_type, max_history> stride_history{};
    std::array<champsim::block_number, max_recent_addr> recent_addr{};
    int last_index = -1;
    int last_action = -1;
    bool stride_history_valid = false;
    int stride_count = 0;
    
    auto index() const
    {
      using namespace champsim::data::data_literals;
      return ip.slice_upper<2_b>();
    }
    auto tag() const
    {
      using namespace champsim::data::data_literals;
      return ip.slice_upper<2_b>();
    }

};
    static constexpr int STRIDE_MIN = -5;
    static constexpr int STRIDE_MAX = 5;
    static constexpr int NUM_STRIDE_BINS = STRIDE_MAX - STRIDE_MIN + 1;
    static constexpr int QTABLE_SIZE = NUM_STRIDE_BINS * NUM_STRIDE_BINS  * NUM_STRIDE_BINS * NUM_STRIDE_BINS * NUM_STRIDE_BINS;

    std::array<std::array<float, MAX_ACTIONS>, QTABLE_SIZE> q_table;
    champsim::msl::lru_table<RLState> table;

    constexpr int stride_to_bin(champsim::block_number::difference_type stride);
    constexpr int encode_state(champsim::block_number::difference_type stride, const std::array<champsim::block_number::difference_type, max_history>& stride_history);     
struct PrefetchTask {
    champsim::address ip;
    champsim::address addr;
    int action;
    int stride; // for stride-based prefetching
};
std::deque<PrefetchTask> prefetch_queue;

enum PrefetchActions{
    no_prefetch = 0,
    simple_stride = 1,
    multi_stride = 2,
    locality = 3,
    correlation = 4
};

    bool issue_stride_prefetch(champsim::address addr, int stride, int degree = 1); // further implementation after the action is decided 
    bool issue_multi_stride_prefetch(champsim::address ip, champsim::address addr, int degree = 1);
    bool issue_locality_prefetch(champsim::address addr, int degree = 2);
    bool issue_correlation_prefetch(champsim::address ip, champsim::address addr);

struct RecentPrefetchHistory{
    static constexpr int HISTORY_SIZE = 64;
    champsim::block_number history[HISTORY_SIZE];   // Stores line addresses
    int head = 0;                     // Circular index

    bool contains(champsim::block_number addr) {
        for (int i = 0; i < HISTORY_SIZE; ++i)
            if (history[i] == addr) return true;
        return false;
    }
    void insert(champsim::block_number addr) {
        history[head] = addr;
        head = (head + 1) % HISTORY_SIZE;
    }
};

struct PrefetchStats
{
  int stride_calls = 0;
  int m_stride_calls = 0;
  int locality_calls = 0;
  int corellation_calls = 0;
};

  private:
    // Helper Functions
    CACHE* cache;
    int select_action(const std::array<float, MAX_ACTIONS>& q_values, float epsilon);  // select actions 
    float compute_reward(access_type type, uint8_t cache_hit, bool useful_prefetch, int action);   // compute reward based on useful prefetch
    void update_q_value(int prev_index, int prev_action, float reward, int curr_index, float alpha = 0.1f, float gamma = 0.9f);
};
#endif
