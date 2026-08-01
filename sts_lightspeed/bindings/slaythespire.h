//
// Created by keega on 9/24/2021.
//

#ifndef STS_LIGHTSPEED_SLAYTHESPIRE_H
#define STS_LIGHTSPEED_SLAYTHESPIRE_H

#include <vector>
#include <unordered_map>
#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "constants/Rooms.h"
#include "constants/Cards.h"
#include "constants/Relics.h"
#include "constants/Potions.h"
#include "game/Card.h"
#include "game/Deck.h"
#include "game/RelicContainer.h"
#include "combat/BattleContext.h"
#include "sim/search/Action.h"

namespace sts {

    struct NNInterface {
        static constexpr int observation_space_size = 412;
        static constexpr int playerHpMax = 200;
        static constexpr int playerGoldMax = 1800;
        static constexpr int cardCountMax = 7;

        const std::vector<int> cardEncodeMap;
        const std::unordered_map<MonsterEncounter, int> bossEncodeMap;

        static inline NNInterface *theInstance = nullptr;

        NNInterface();

        int getCardIdx(Card c) const;
        std::array<int,observation_space_size> getObservationMaximums() const;
        std::array<int,observation_space_size> getObservation(const GameContext &gc) const;


        static std::vector<int> createOneHotCardEncodingMap();
        static std::unordered_map<MonsterEncounter, int> createBossEncodingMap();
        static NNInterface* getInstance();

    };

    namespace search {
        class ScumSearchAgent2;
    }


    class GameContext;
    class Map;

    namespace py {

        // Compatibility representation used by Silverbot's ModelHP checkpoints.
        // This deliberately coexists with NNInterface's legacy flat observation;
        // Heart1 was trained on this structured representation.
        template<typename T> pybind11::array_t<T> to_numpy(const std::vector<T>& vec) {
            auto result = pybind11::array_t<T>(vec.size());
            auto r = result.template mutable_unchecked<1>();
            for (pybind11::ssize_t i = 0; i < vec.size(); ++i) r(i) = vec[i];
            return result;
        }

        static constexpr int fixed_observation_space_size = 10;
        static constexpr int fixed_player_hp_max = 200;
        static constexpr int fixed_player_gold_max = 1800;
        static constexpr int fixed_num_bosses = 10;

        struct NNCardsRepresentation {
            pybind11::array_t<CardId> cards;
            pybind11::array_t<int> upgrades;
            pybind11::dict as_dict() const;
        };
        struct NNRelicsRepresentation {
            pybind11::array_t<RelicId> relics;
            pybind11::array_t<int> relicCounters;
            pybind11::dict as_dict() const;
        };
        struct NNMapRepresentation {
            pybind11::array_t<int> xs;
            pybind11::array_t<int> ys;
            pybind11::array_t<Room> roomTypes;
            pybind11::array_t<int> pathXs;
            int burningEliteX = -1;
            int burningEliteY = -1;
            pybind11::dict as_dict() const;
        };
        struct NNRepresentation {
            pybind11::array_t<int> fixedObservation;
            NNCardsRepresentation deck;
            NNRelicsRepresentation relics;
            pybind11::array_t<Potion> potions;
            NNMapRepresentation map;
            int mapX = -1, mapY = -1;
            pybind11::dict as_dict() const;
        };

        void play();

        search::ScumSearchAgent2* getAgent();
        void setGc(const GameContext &gc);
        GameContext* getGc();

        void playout();
        std::vector<Card> getCardReward(GameContext &gc);
        void pickRewardCard(GameContext &gc, Card card);
        void skipRewardCards(GameContext &gc);

        std::vector<int> getNNMapRepresentation(const Map &map);
        Room getRoomType(const Map &map, int x, int y);
        bool hasEdge(const Map &map, int x, int y, int x2);

        pybind11::array_t<int> getFixedObservationMaximums();
        pybind11::array_t<int> getFixedObservation(const GameContext &gc);
        NNCardsRepresentation getCardRepresentation(const Deck &deck);
        NNRelicsRepresentation getRelicRepresentation(const RelicContainer &relics);
        NNMapRepresentation getStructuredNNMapRepresentation(const Map &map);
        NNRepresentation getNNRepresentation(const GameContext &gc);

        // Isolated single-fight interface for RL training against just combat
        // (not a full run). Added on top of the existing engine rather than
        // modifying BattleScumSearcher2's own enumeration methods, to avoid
        // touching code the built-in search agent depends on.
        BattleContext newBattle(GameContext &gc, MonsterEncounter encounter);
        std::vector<search::Action> getLegalActions(const BattleContext &bc);
        std::pair<int,int> getMonsterMoveDamage(const BattleContext &bc, int monsterIdx);

        // Generic name-based PlayerStatus lookup (Player::getStatusRuntime
        // already handles the "not present" -> 0 case safely) -- avoids
        // declaring the ~90-entry PlayerStatus enum in pybind just to read a
        // handful of power stacks (Artifact, Metallicize, ...) from Python.
        // Throws if `name` doesn't match any entry in playerStatusEnumStrings
        // (a typo here is a programming error, not a runtime game state).
        int getPlayerStatusValue(const BattleContext &bc, const std::string &name);

        // Name-based MonsterStatus lookup for a single monster. Unlike the
        // player version this uses an explicit name->templated-getStatus<>
        // switch rather than an enum-ordinal index into
        // monsterStatusEnumStrings, because that string array is NOT 1:1
        // with the MonsterStatus enum (REACTIVE/SHARP_HIDE are reordered) --
        // indexing it by ordinal would read the wrong field. Only the
        // handful of statuses the RL observation actually needs are wired
        // (TIME_WARP, POISON, PLATED_ARMOR, ARTIFACT, METALLICIZE,
        // MODE_SHIFT, INTANGIBLE, CURL_UP); throws on any other name so a
        // typo or an un-wired status fails loudly instead of silently
        // returning 0. Returns 0 for an out-of-range monsterIdx.
        int getMonsterStatusValue(const BattleContext &bc, int monsterIdx, const std::string &name);

        // Raw Monster::miscInfo passthrough -- NOT an MS:: status, a
        // per-monster-type-specific int (see bindings-util.cpp's own
        // comment). For Time Eater, nonzero means "already used Haste this
        // fight" (its one-time <=50%-HP heal-and-clear-all-debuffs proc);
        // for other monster types this field means something else entirely,
        // so callers must gate on monster identity, not treat this as a
        // universal signal. Returns 0 for an out-of-range monsterIdx.
        int getMonsterMiscInfo(const BattleContext &bc, int monsterIdx);

        // Classifies monster `monsterIdx`'s CURRENT queued move by actually
        // running it -- on a throwaway BattleContext(bc) copy, never bc
        // itself -- and observing what changed, rather than a hand-built
        // MonsterMoveId -> category lookup table (196 move ids across ~47
        // monster types would mean guessing at moves never individually
        // verified). Monster::takeTurn(bc) is self-contained (dispatches on
        // its own moveHistory[0]/idx, not on any external "whose turn is
        // it" index), so isolating one monster's move this way is safe and
        // exact -- confirmed via a real test (Donu's Buff move correctly
        // reports buffs_ally=true). Damage is NOT re-derived here --
        // getMonsterMoveDamage already answers that cleanly; this only
        // covers the non-damage categories a pure damage check can't:
        // self-buff (Strength/Artifact/Plated Armor/Metallicize increased),
        // ally-buff (another living monster's Strength increased -- Donu &
        // Deca's whole mechanic), and player-debuff (Weak/Vulnerable/Frail
        // increased, or Strength/Dexterity reduced) with no damage
        // attached. Returns all-false for an out-of-range/dead monsterIdx.
        struct MoveCategory {
            bool self_buffs = false;
            bool buffs_ally = false;
            bool debuffs_player = false;
        };
        MoveCategory classifyMonsterMove(const BattleContext &bc, int monsterIdx);
    }


}


#endif //STS_LIGHTSPEED_SLAYTHESPIRE_H
