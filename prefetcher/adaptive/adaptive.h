#ifndef ADAPTIVE_H
#define ADAPTIVE_H

#include <deque>
#include <unordered_map>
#include <vector>
#include "address.h"
#include "champsim.h"
#include "modules.h"
#include "msl/lru_table.h"

enum PrefetchAction : int8_t {
    NO_PREFETCH = 0,
    STRIDE_PREFETCH = 1,
    MULTI_STRIDE_PREFETCH = 2,
    LOCALITY_PREFETCH = 3,
    CORRELATION_PREFETCH = 4
};

constexpr int MAX_ACTIONS = 5;

struct adaptive : public champsim::modules::prefetcher {

    explicit adaptive(CACHE* cache);
    // Prefetcher API Hooks
    uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type, uint32_t metadata_in);
    uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);
    void prefetcher_cycle_operate();

    constexpr static std::size_t Q_TABLE_SETS = 512;
    constexpr static std::size_t Q_TABLE_SET_ENTRIES = 8;
    constexpr static std::size_t TABLE_SETS = 256;   // There are 256 sets (buckets).
    constexpr static std::size_t TABLE_SET_ENTRIES = 4; // Each set can store up to 4 entries.
    // When a 5th IP maps to the same set, one of the 4 entries is evicted (typically Least Recently Used — LRU).

struct RLState {
    static constexpr size_t max_history = 4;
    static constexpr size_t max_recent_addr = 8;

    champsim::address ip{};
    champsim::block_number last_cl_addr{};
    champsim::block_number::difference_type last_stride{};
    std::array<champsim::block_number::difference_type, max_history> stride_history{};
    std::array<champsim::block_number, max_recent_addr> recent_addr{};

    std::vector<int> active_tiles() const;
    int last_action;

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

struct QTableEntry {
    std::vector<int> key;                            // Active tile indices (state encoding)
    std::array<float, MAX_ACTIONS> q_values{};       // One Q-value per action

    // Determines which set (bucket) this entry maps to.
    std::size_t index() const {
        std::size_t idx = 0;
        for (size_t i = 0; i < key.size(); ++i)
            idx += static_cast<std::size_t>(key[i]) * (i + 1);
        return idx % Q_TABLE_SETS;
    }
    std::vector<int> tag() const {
        return key;
    }
    bool operator==(const QTableEntry& other) const {
        return key == other.key;
    }
};
    champsim::msl::lru_table<QTableEntry> q_table_lru;
    champsim::msl::lru_table<RLState> table;

     
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

    bool issue_stride_prefetch(champsim::address addr, int stride, int degree = 2); // further implementation after the action is decided 
    bool issue_multi_stride_prefetch(champsim::address ip, champsim::address addr, int degree = 3);
    bool issue_locality_prefetch(champsim::address addr, int degree = 2);
    bool issue_correlation_prefetch(champsim::address ip, champsim::address addr);

  private:
    // Helper Functions
    CACHE* cache;
    int select_action(const std::array<float, MAX_ACTIONS>& q_values, float epsilon);  // select actions 
    float compute_reward(access_type type, uint8_t cache_hit, bool useful_prefetch);   // compute reward based on useful prefetch
    void update_q_value(const QTableEntry& prev_entry, int prev_action, float reward, const QTableEntry& curr_entry, 
                        const std::array<float, MAX_ACTIONS>& curr_q_values, float alpha = 0.1f, float gamma = 0.9f); // update the q-value with reward
};
#endif
