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

    adaptive(CACHE* cache);

    // Prefetcher API Hooks
    uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type, uint32_t metadata_in);
    // Called every time the CPU accesses the cache. Can be used to record data
    uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);
    // Called when a block (line) is inserted into the cache. Can be demand miss or a prefetch was successful
    void prefetcher_cycle_operate();
    // Called every cycle, regardless of accesses.



struct RLState {
    static constexpr size_t max_history = 4;
    static constexpr size_t max_recent_blocks = 8;

    champsim::address ip{};
    champsim::block_number last_cl_addr{};
    champsim::block_number::difference_type last_stride{};
    std::array<champsim::block_number::difference_type, max_history> stride_history{};
    std::array<uint64_t, max_recent_blocks> access_timestamps{};

    std::vector<int> active_tiles() const;

    access_type last_access_type{};
    bool was_cache_hit = false;
    uint8_t prefetch_count = 0;
    float last_prefetch_useful = 0.0;
    size_t total_prefetches = 0;
    size_t useful_prefetches = 0;
};

struct RLPatterns{
    float confidence_simple_stride = 0.0f;
    float confidence_multi_stride = 0.0f;
    float confidence_locality = 0.0f;
    float confidence_correlation = 0.0f;
};

struct QTableEntry {
    std::vector<int> key;                            // Active tile indices
    std::array<float, MAX_ACTIONS> q_values{};       // One Q-value per action

    size_t index() const {
        // Hash to determine which set this goes into 
        size_t hash = 0;
        for (int val : key)
            hash ^= std::hash<int>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
    /* Similar to boosts hash combining
    | Part                    | Meaning                                                     |
    | ----------------------- | ----------------------------------------------------------- |
    | `std::hash<int>{}(val)` | Hashes individual tile index                                |
    | `0x9e3779b9`            | Golden ratio constant in hex (≈ 2^32 / φ) to spread entropy | magic number from Fibonacci hashing(inverse of golden ratio)
    | `hash << 6` and `>> 2`  | Bit-shifts to mix up bits                                   |
    | `^=` (XOR assignment)   | Combines the new hashed tile with the cumulative hash       |
    */

    std::vector<int> tag() const {
        return key; // Match full key for exact hit
    }

    // Required for comparison in tag() logic
    bool operator==(const QTableEntry& other) const {
        return key == other.key;
    }
};
    constexpr static std::size_t Q_TABLE_SETS = 512;
    constexpr static std::size_t Q_TABLE_SET_ENTRIES = 8;
    constexpr static std::size_t TABLE_SETS = 256;   // There are 256 sets (buckets).
    constexpr static std::size_t TABLE_SET_ENTRIES = 4; // Each set can store up to 4 entries.
    // When a 5th IP maps to the same set, one of the 4 entries is evicted (typically Least Recently Used — LRU).

    champsim::msl::lru_table<QTableEntry> q_table_lru{Q_TABLE_SETS, Q_TABLE_SET_ENTRIES};
    champsim::msl::lru_table<RLState> table{TABLE_SETS, TABLE_SET_ENTRIES};
    
    std::unordered_map<champsim::block_number, champsim::block_number> correlation_table;

    // Helper Functions
    void update_confidence(float& conf, bool detected, bool useful);
    int select_action(const std::array<float, MAX_ACTIONS>& q_values, float epsilon);

  private:
    void update_reward(const RLState& state, bool useful);
    void train_q_table(const RLState& state, int8_t reward);
};
#endif
