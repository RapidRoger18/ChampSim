#include <array>
#include <cmath>
#include <optional>

#include "bandit.h"
#include "cache.h"


  bandit::bandit(CACHE* cache) : champsim::modules::prefetcher(cache), stride(cache), ampmlite(cache) {}

int bandit::select_prefetcher() {
    double best_score = -1e9;
    int best_action = 0;
  
    for (int i = 0; i < NUM_PREFETCHERS; ++i) {
      double bonus = std::sqrt(std::log(totalswitches) / nTable[i]);
      double score = rTable[i] + bonus;
      if (score > best_score) {
        best_score = score;
        best_action = i;
      }
    }
  
    return best_action;
}
void bandit::update_reward(double reward) {
    rTable[current_pref] = 0.9 * rTable[current_pref] + 0.1 * reward;
    nTable[current_pref]++;
    totalswitches++;
}

uint32_t bandit::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type, uint32_t metadata_in){
  access_counter++;
  if (access_counter % SWITCH_PREFETCHER == 0) {
    update_reward(1.0);
    current_pref = select_prefetcher();
  }
  switch (current_pref) {
    case 0:
      return stride.prefetcher_cache_operate(addr, ip, cache_hit, useful_prefetch, type, metadata_in);
    case 1:
      return ampmlite.prefetcher_cache_operate(addr, ip, cache_hit, useful_prefetch, type, metadata_in);
    default:
      return metadata_in;
  }
}

uint32_t bandit::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void bandit::prefetcher_cycle_operate()
{
  switch (current_pref) {
    case 0: stride.prefetcher_cycle_operate(); break;
    default: break;
  }
}