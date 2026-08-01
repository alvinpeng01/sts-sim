# AGENTS.md

C++17 Slay the Spire engine. Forked from gamerpuppy/sts_lightspeed. Pybind11 bindings expose it as the Python `slaythespire` module.

## OVERVIEW

Multi-class card implementations (Ironclad, Silent, Defect, Watcher), full overworld GameContext,
expectimax MCTS combat search (55 tunable parameters), deterministic RNG. The bindings layer
(slaythespire.cpp, 5,310 LOC) is the AUTHORITATIVE runtime. Python expectimax_search.py exists
only as a debugging aid; the real search runs in C++. Heuristic classifiers are character-agnostic:
`isAoeCard`, `isVulnerableApplier`, `isDefensiveCard`, `isSilentPoisonApplier`, and
`nativeImmediateBlockBase` cover all 4 classes.

## STRUCTURE

```
src/combat/       BattleContext, Actions, Player, Monster, MonsterGroup,
                  MonsterSpecific, CardInstance, CardManager, CardQueue
src/game/         GameContext (overworld state machine), Game, Deck, Map,
                  Shop, Neow, Card, SaveFile, CombatReward
src/sim/          BattleSimulator, ConsoleSimulator, PrintHelpers
src/sim/search/   MCTS agents: Action (32-bit packed), SimpleAgent
                  (heuristic rollout), ScumSearchAgent2, ExpertKnowledge, GameAction
include/          50 headers + constants/ (Cards, Relics, Monsters, Events,
                  Potions, Rooms — 15 data tables)
bindings/         slaythespire.cpp, slaythespire.h, bindings-util.cpp
apps/             main.cpp (interactive), test.cpp (benchmarks + MCTS),
                  small-test.cpp (validation)
json/             nlohmann/json (vendored submodule)
pybind11/         pybind11 (vendored submodule)
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Python bindings (all) | bindings/slaythespire.cpp |
| Native MCTS + heuristics | bindings/slaythespire.cpp (nativeRunMctsSearch, nativeHeuristicPickFast, nativeExpectimaxTerminalReward, nativePredictedIncomingDamage) |
| TunableParams struct (55 params) | bindings/slaythespire.cpp:120-332 |
| Overworld state machine | src/game/GameContext.cpp |
| Battle engine | src/combat/BattleContext.cpp |
| Combat actions | src/combat/Actions.cpp |
| Search actions (32-bit packed) | src/sim/search/Action.cpp |
| Heuristic rollout (133-card priority) | src/sim/search/SimpleAgent.cpp |
| Expert knowledge | src/sim/search/ExpertKnowledge.cpp |
| Card data tables | include/constants/Cards.h |
| Monster data | include/constants/MonsterIds.h, MonsterEncounters.h |
| Relic data | include/constants/Relics.h |
| CMake config | CMakeLists.txt |

## CONVENTIONS

- CMake 3.19+ required. Compiler flags: `-O3 -DNDEBUG -flto -Wno-shift-count-overflow` (MinGW GCC)
- Naming: PascalCase classes, camelCase methods, UPPER_SNAKE_CASE enums
- Fixed-size containers via fixed_list.h for performance (avoid std::vector hot paths)
- Macro-based codegen: FOREACH_ACTIONTYPE in Actions.h drives switch-case dispatchers
- Pybind11 bindings use `.def_readwrite("pythonName", &CppClass::cppFieldName)` pattern
- Nested namespaces must be flattened for pybind11: Neow::Bonus → NeowBonus
- DRIFT WARNING comments in slaythespire.cpp mark sections that must stay in sync with specific Python originals
- All 55 tunable MCTS params live in TunableParams struct, all exposed to Python via `sts.set_search_params()` in snake_case. `slay-sim/lightspeed/tuned_search_params.json` overrides 29; the other 26 sit at compiled defaults
- The shipped config enables SEQUENTIAL HALVING at the root (`fitness_config.seq_halving`), so `nativeRunMctsSearchSeqHalving` — which picks by mean value, not visit count — is the production driver
- `make -j8` rebuild required after ANY header or binding change (no incremental bindings)

## ANTI-PATTERNS

- NEVER assume RNG parity with base game — battle RNG streams were unified into single stream
- Do NOT add new CardSelectTask types without wiring BOTH Action.cpp AND BattleSimulator.cpp (two separate dispatchers)
- Enum bounds: IntEnum classes exposed via pybind11 must start at 0 (EnumSpace checks `0 <= x < len`)
- Object access in Python: `gc.deck[idx]` NOT `gc.deck.cards[idx]`; `gc.relics[idx].id` NOT `gc.relics[idx].getId()`
- BattleScumSearcher2 is OLD search (uniform random rollouts), NOT used. nativeRunMctsSearch is current.
- Do NOT treat expectimax_search.py as runtime truth — slaythespire.cpp is authoritative
- NEVER trust `getCardColor()` / `cardColors[]` (`include/constants/Cards.h:424`) as a card's character — 8 entries are wrong (Brilliance, Collect, Brutality, Combust, Buffer, Compile Driver, Bullet Time, Concentrate); known and deferred, see `../docs/07-known-issues.md`

## COMMANDS

```bash
# First-time build
cd sts_lightspeed && mkdir build && cd build && cmake .. -G "MinGW Makefiles"
cmake --build . --target slaythespire -j8

# Rebuild after binding/header changes
cd sts_lightspeed/build && cmake --build . --target slaythespire -j8

# Run C++ tests (agent_mt mode: 4 threads, 2 iters, seed 12345, 1000 depth, 1 batch)
cd sts_lightspeed/build && ./test.exe agent_mt 4 2 20 12345 1000 1

# Verify Python import
cd slay-sim && $env:PYTHONPATH='..\sts_lightspeed\build;.' ; python -c "import slaythespire; print('ok')"
```
