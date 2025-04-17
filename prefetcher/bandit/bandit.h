#ifndef BANDIT_H
#define BANDIT_H

#include "address.h"
#include "champsim.h"
#include "modules.h"

#include "../ip_stride/ip_stride.h"
#include "../va_ampm_lite/va_ampm_lite.h"
#include "../spp_dev/spp_dev.h"
#include "../next_line/next_line.h"


struct bandit : public champsim::modules::prefetcher {
    static constexpr int NUM_PREFETCHERS = 2;
    static constexpr int SWITCH_PREFETCHER = 1000; 
  
    int current_pref = 0;
    int access_counter = 0;
    int totalswitches = 1;
    
    std::array<double, NUM_PREFETCHERS> rTable = {1.0, 1.0};
    std::array<int, NUM_PREFETCHERS> nTable = {1, 1};

    ip_stride stride;
    va_ampm_lite ampmlite;
    
    bandit(CACHE* cache);
    // inbuilt function calls
    uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                      uint32_t metadata_in);
    uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);
    void prefetcher_cycle_operate();
  
  private:
    int select_prefetcher();
    void update_reward(double reward);
  };
#endif