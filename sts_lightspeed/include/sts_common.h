//
// Created by gamerpuppy on 9/4/2021.
//

#ifndef STS_LIGHTSPEED_STS_COMMON_H
#define STS_LIGHTSPEED_STS_COMMON_Hs
//#define sts_print_debug
#define sts_asserts

//#define sts_action_queue_use_raw_array
//#define sts_fixed_list_use_raw_array
//#define sts_card_manager_use_fixed_list
// Left disabled. Investigated at length (see conversation/commit history):
// enabling this crashes fixed_list.h's bounds-check assert during MCTS search
// against DONU_AND_DECA at every capacity tried (64 up through 400), even
// after adding NATIVE_MAX_TURNS_PER_SEARCH (slaythespire.cpp) to bound
// simulated fight length. Direct instrumentation -- peak draw+discard+
// exhaust+hand size, tracked at EVERY BattleContext creation site (rollout
// copies AND the tree's own persistent node states, plus the real/root
// battle state itself) -- never showed a peak above 64 across dozens of
// repeated trials of the exact crashing seed, even on a full clean rebuild
// (ruling out a stale-incremental-build/ABI-mismatch explanation). That is a
// genuine, unresolved contradiction: the assert fires well before any
// diagnostic-visible size approaches even the smallest capacity tried. This
// suggests the bug may not be "real card count too large" at all, but
// something subtler in how fixed_list's specific semantics (its resize()/
// remove() do NOT clear the now-unused array slots, unlike std::vector)
// interact with some card-management code path -- e.g. list_size itself
// reading corrupted/stale rather than genuinely growing. Pinning this down
// needs a real debugger or memory sanitizer (breakpoint on the assert,
// inspect the actual fixed_list's memory) rather than further printf-style
// tracing. NATIVE_MAX_TURNS_PER_SEARCH itself is unrelated and safe -- kept.


#include <cstdint>



#endif //STS_LIGHTSPEED_STS_COMMON_H
