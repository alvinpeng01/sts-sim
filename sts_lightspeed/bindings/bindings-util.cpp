//
// Created by keega on 9/24/2021.
//
#include <sstream>
#include <algorithm>

#include "sim/ConsoleSimulator.h"
#include "sim/search/ScumSearchAgent2.h"
#include "sim/SimHelpers.h"
#include "sim/PrintHelpers.h"
#include "game/Game.h"
#include "game/Map.h"

#include "slaythespire.h"

namespace sts {

    NNInterface::NNInterface() :
            cardEncodeMap(createOneHotCardEncodingMap()),
            bossEncodeMap(createBossEncodingMap()) {}

    int NNInterface::getCardIdx(Card c) const {
        int idx = cardEncodeMap[static_cast<int>(c.id)] * 2;
        if (idx == -1) {
            std::cerr << "attemped to get encoding idx for invalid card" << std::endl;
            assert(false);
        }

        if (c.isUpgraded()) {
            idx += 1;
        }

        return idx;
    }

    std::array<int,NNInterface::observation_space_size> NNInterface::getObservation(const GameContext &gc) const {
        std::array<int,observation_space_size> ret {};

        int offset = 0;

        ret[offset++] = std::min(gc.curHp, playerHpMax);
        ret[offset++] = std::min(gc.maxHp, playerHpMax);
        ret[offset++] = std::min(gc.gold, playerGoldMax);
        ret[offset++] = gc.floorNum;

        int bossEncodeIdx = offset + bossEncodeMap.at(gc.boss);
        ret[bossEncodeIdx] = 1;
        offset += 10;

        for (auto c : gc.deck.cards) {
            int encodeIdx = offset + getCardIdx(c);
            ret[encodeIdx] = std::min(ret[encodeIdx]+1, cardCountMax);
        }
        offset += 220;

        for (auto r : gc.relics.relics) {
            int encodeIdx = offset + static_cast<int>(r.id);
            ret[encodeIdx] = 1;
        }
        offset += 178;

        return ret;
    }

    std::array<int,NNInterface::observation_space_size> NNInterface::getObservationMaximums() const {
        std::array<int,observation_space_size> ret {};
        int spaceOffset = 0;

        ret[0] = playerHpMax;
        ret[1] = playerHpMax;
        ret[2] = playerGoldMax;
        ret[3] = 60;
        spaceOffset += 3;

        std::fill(ret.begin()+spaceOffset, ret.end(), 1);
        spaceOffset += 10;

        std::fill(ret.begin()+spaceOffset, ret.end(), cardCountMax);
        spaceOffset += 220;

        std::fill(ret.begin()+spaceOffset, ret.end(), 1);
        spaceOffset += 178;

        return ret;
    }

    std::vector<int> NNInterface::createOneHotCardEncodingMap() {
        std::vector<CardId> redCards;
        for (int i = static_cast<int>(CardId::INVALID); i <= static_cast<int>(CardId::ZAP); ++i) {
            auto cid = static_cast<CardId>(i);
            auto color = getCardColor(cid);
            if (color == CardColor::RED) {
                redCards.push_back(cid);
            }
        }

        std::vector<CardId> colorlessCards;
        for (int i = 0; i < srcColorlessCardPoolSize; ++i) {
            colorlessCards.push_back(srcColorlessCardPool[i]);
        }
        std::sort(colorlessCards.begin(), colorlessCards.end(), [](auto a, auto b) {
            return std::string(getCardEnumName(a)) < std::string(getCardEnumName(b));
        });

        std::vector<int> encodingMap(372);
        std::fill(encodingMap.begin(), encodingMap.end(), 0);

        int hotEncodingIdx = 0;
        for (auto x : redCards) {
            encodingMap[static_cast<int>(x)] = hotEncodingIdx++;
        }
        for (auto x : colorlessCards) {
            encodingMap[static_cast<int>(x)] = hotEncodingIdx++;
        }

        return encodingMap;
    }

    std::unordered_map<MonsterEncounter, int> NNInterface::createBossEncodingMap() {
        std::unordered_map<MonsterEncounter, int> bossMap;
        bossMap[ME::SLIME_BOSS] = 0;
        bossMap[ME::HEXAGHOST] = 1;
        bossMap[ME::THE_GUARDIAN] = 2;
        bossMap[ME::CHAMP] = 3;
        bossMap[ME::AUTOMATON] = 4;
        bossMap[ME::COLLECTOR] = 5;
        bossMap[ME::TIME_EATER] = 6;
        bossMap[ME::DONU_AND_DECA] = 7;
        bossMap[ME::AWAKENED_ONE] = 8;
        bossMap[ME::THE_HEART] = 9;
        return bossMap;
    }

    NNInterface* NNInterface::getInstance() {
        if (theInstance == nullptr) {
            theInstance = new NNInterface;
        }
        return theInstance;
    }

}

namespace sts::py {

    void play() {
        sts::SimulatorContext ctx;
        sts::ConsoleSimulator sim;
        sim.play(std::cin, std::cout, ctx);
    }

    search::ScumSearchAgent2* getAgent() {
        static search::ScumSearchAgent2 *agent = nullptr;
        if (agent == nullptr) {
            agent = new search::ScumSearchAgent2();
            agent->pauseOnCardReward = true;
        }
        return agent;
    }

    void playout(GameContext &gc) {
        auto agent = getAgent();
        agent->playout(gc);
    }

    std::vector<Card> getCardReward(GameContext &gc) {
        const bool inValidState = gc.outcome == GameOutcome::UNDECIDED &&
                                  gc.screenState == ScreenState::REWARDS &&
                                  gc.info.rewardsContainer.cardRewardCount > 0;

        if (!inValidState) {
            std::cerr << "GameContext was not in a state with card rewards, check that the game has not completed first." << std::endl;
            return {};
        }

        const auto &r = gc.info.rewardsContainer;
        const auto &cardList = r.cardRewards[r.cardRewardCount-1];
        return std::vector<Card>(cardList.begin(), cardList.end());
    }

    void pickRewardCard(GameContext &gc, Card card) {
        const bool inValidState = gc.outcome == GameOutcome::UNDECIDED &&
                                  gc.screenState == ScreenState::REWARDS &&
                                  gc.info.rewardsContainer.cardRewardCount > 0;
        if (!inValidState) {
            std::cerr << "GameContext was not in a state with card rewards, check that the game has not completed first." << std::endl;
            return;
        }
        auto &r = gc.info.rewardsContainer;
        gc.deck.obtain(gc, card);
        r.removeCardReward(r.cardRewardCount-1);
    }

    void skipRewardCards(GameContext &gc) {
        const bool inValidState = gc.outcome == GameOutcome::UNDECIDED &&
                                  gc.screenState == ScreenState::REWARDS &&
                                  gc.info.rewardsContainer.cardRewardCount > 0;
        if (!inValidState) {
            std::cerr << "GameContext was not in a state with card rewards, check that the game has not completed first." << std::endl;
            return;
        }

        if (gc.hasRelic(RelicId::SINGING_BOWL)) {
            gc.playerIncreaseMaxHp(2);
        }

        auto &r = gc.info.rewardsContainer;
        r.removeCardReward(r.cardRewardCount-1);
    }

    // ---- Silverbot Heart1 observation compatibility --------------------
    // Keep this separate from NNInterface's old 412-wide one-hot vector:
    // Heart1's checkpoint was trained on the structured, variable-length
    // representation below.
    pybind11::array_t<int> getFixedObservation(const GameContext &gc) {
        std::vector<int> ret(fixed_observation_space_size);
        int offset = 0;
        ret[offset++] = std::min(gc.curHp, fixed_player_hp_max);
        ret[offset++] = std::min(gc.maxHp, fixed_player_hp_max);
        ret[offset++] = std::min(gc.gold, fixed_player_gold_max);
        ret[offset++] = gc.floorNum;
        ret[offset++] = NNInterface::getInstance()->bossEncodeMap.at(gc.boss);
        ret[offset++] = gc.info.toSelectCount;
        ret[offset++] = gc.ascension;
        ret[offset++] = gc.redKey ? 1 : 0;
        ret[offset++] = gc.greenKey ? 1 : 0;
        ret[offset++] = gc.blueKey ? 1 : 0;
        return to_numpy(ret);
    }

    pybind11::array_t<int> getFixedObservationMaximums() {
        return to_numpy(std::vector<int>{fixed_player_hp_max, fixed_player_hp_max,
            fixed_player_gold_max, 56, fixed_num_bosses, Deck::MAX_SIZE, 20, 1, 1, 1});
    }

    NNCardsRepresentation getCardRepresentation(const Deck &deck) {
        std::vector<CardId> cards;
        std::vector<int> upgrades;
        for (int i = 0; i < deck.size(); ++i) {
            cards.push_back(deck.cards[i].id);
            upgrades.push_back(deck.cards[i].getUpgraded());
        }
        return {to_numpy(cards), to_numpy(upgrades)};
    }

    NNRelicsRepresentation getRelicRepresentation(const RelicContainer &relics) {
        std::vector<RelicId> ids;
        std::vector<int> counters;
        for (int i = 0; i < relics.size(); ++i) {
            ids.push_back(relics.relics[i].id);
            counters.push_back(relics.relics[i].data);
        }
        return {to_numpy(ids), to_numpy(counters)};
    }

    NNMapRepresentation getStructuredNNMapRepresentation(const Map &map) {
        std::vector<int> xs, ys;
        std::vector<Room> roomTypes;
        std::vector<std::array<int, 3>> paths;
        bool haveLastRow = false;
        for (int y = 0; y < 15; ++y) {
            for (int x = 0; x < 7; ++x) {
                const MapNode &node = map.getNode(x, y);
                if (node.room == Room::NONE) continue;
                xs.push_back(x); ys.push_back(y); roomTypes.push_back(node.room);
                haveLastRow = haveLastRow || y == 14;
                std::array<int, 3> row{-1, -1, -1};
                for (int i = 0; i < node.edgeCount; ++i) {
                    const int edgeX = node.edges[i];
                    if (y == 14) row[1] = edgeX;
                    else if (edgeX == x - 1) row[0] = edgeX;
                    else if (edgeX == x) row[1] = edgeX;
                    else if (edgeX == x + 1) row[2] = edgeX;
                }
                paths.push_back(row);
            }
        }
        if (haveLastRow) {
            xs.push_back(3); ys.push_back(15); roomTypes.push_back(Room::BOSS);
            paths.push_back({-1, -1, -1});
        }
        auto pathArray = pybind11::array_t<int>(std::vector<pybind11::ssize_t>{
            static_cast<pybind11::ssize_t>(paths.size()), 3});
        auto accessor = pathArray.mutable_unchecked<2>();
        for (pybind11::ssize_t i = 0; i < static_cast<pybind11::ssize_t>(paths.size()); ++i)
            for (pybind11::ssize_t j = 0; j < 3; ++j) accessor(i, j) = paths[i][j];
        return {to_numpy(xs), to_numpy(ys), to_numpy(roomTypes), pathArray,
                map.burningEliteX, map.burningEliteY};
    }

    NNRepresentation getNNRepresentation(const GameContext &gc) {
        NNRepresentation rep;
        rep.fixedObservation = getFixedObservation(gc);
        rep.deck = getCardRepresentation(gc.deck);
        rep.relics = getRelicRepresentation(gc.relics);
        std::vector<Potion> potions;
        for (int i = 0; i < gc.potionCapacity; ++i) potions.push_back(gc.potions[i]);
        rep.potions = to_numpy(potions);
        rep.map = getStructuredNNMapRepresentation(*gc.map);
        rep.mapX = gc.curMapNodeX;
        rep.mapY = gc.curMapNodeY;
        return rep;
    }



    // BEGIN MAP THINGS ****************************

    std::vector<int> getNNMapRepresentation(const Map &map) {
        std::vector<int> ret;

        // 7 bits
        // push edges to first row
        for (int x = 0; x < 7; ++x) {
            if (map.getNode(x,0).edgeCount > 0) {
                ret.push_back(true);
            } else {
                ret.push_back(false);
            }
        }

        // for each node in a row, push valid edges to next row, 3 bits per node, 21 bits per row
        // skip 14th row because it is invariant
        // 21 * 13 == 273 bits
        for (int y = 0; y < 14; ++y) {
            for (int x = 0; x < 7; ++x) {

                bool localEdgeValues[3] {false, false, false};
                auto node = map.getNode(x,y);
                for (int i = 0; i < node.edgeCount; ++i) {
                    auto edge = node.edges[i];
                    if (edge < x) {
                        localEdgeValues[0] = true;
                    } else if (edge == x) {
                        localEdgeValues[1] = true;
                    } else {
                        localEdgeValues[2] = true;
                    }
                }
                ret.insert(ret.end(), localEdgeValues, localEdgeValues+3);
            }
        }

        // room types - for each node there are 6 possible rooms,
        // the first row is always monster, the 8th row is always treasure, 14th is always rest
        // this gives 14-3 valid rows == 11
        // 11 * 6 * 7 = 462 bits
        for (int y = 1; y < 14; ++y) {
            if (y == 8) {
                continue;
            }
            for (int x = 0; x < 7; ++x) {
                auto roomType = map.getNode(x,y).room;
                for (int i = 0; i < 6; ++i) {
                    ret.push_back(static_cast<int>(roomType) == i);
                }
            }
        }

        return ret;
    };

    Room getRoomType(const Map &map, int x, int y) {
        if (x < 0 || x > 6 || y < 0 || y > 14) {
            return Room::INVALID;
        }

        return map.getNode(x,y).room;
    }

    bool hasEdge(const Map &map, int x, int y, int x2) {
        if (x == -1) {
            return map.getNode(x2,0).edgeCount > 0;
        }

        if (x < 0 || x > 6 || y < 0 || y > 14) {
            return false;
        }


        auto node = map.getNode(x,y);
        for (int i = 0; i < node.edgeCount; ++i) {
            if (node.edges[i] == x2) {
                return true;
            }
        }
        return false;
    }

    // --- isolated single-fight interface (RL training against just combat) ---

    BattleContext newBattle(GameContext &gc, MonsterEncounter encounter) {
        BattleContext bc;
        bc.init(gc, encounter);
        return bc;
    }

    // Mirrors BattleScumSearcher2::enumerateCardActions/enumerateActionsForNode's
    // CARD + END_TURN branch, as a standalone function returning a plain vector
    // instead of populating search tree nodes -- written new rather than
    // reusing the searcher's private methods, to avoid touching code the
    // built-in search agent depends on. CARD_SELECT (post-play choices like
    // Exhume/Warcry/Armaments) IS handled, via Action's own
    // enumerateCardSelectActions -- this was found by testing: a random-policy
    // playout hit an empty legal-actions list the moment it played a
    // card-select-triggering card, which is exactly the gap this closes.
    //
    // POTION actions ARE now handled too (drink + discard, mirroring
    // isValidPotionAction in Action.cpp -- FAIRY_POTION excluded from drink,
    // same as there, since it auto-triggers on death rather than being
    // manually drunk). Still deliberately omitted: BattleContext.h's
    // InputState enum has several OTHER player-choice states some potions/
    // cards can trigger (CHOOSE_STANCE_ACTION from Stance Potion, Watcher-
    // only so unreachable for Ironclad-only training anyway;
    // CHOOSE_GAMBLING_CARDS from Gambler's Brew; CHOOSE_ENTROPIC_BREW_DISCARD_POTIONS
    // from Entropic Brew) -- not wired up here, so a potion that lands one of
    // those states will still hit the same empty-list failure mode this
    // function's docstring used to describe for potions generally. The
    // Python-side potion pool (lightspeed/potions.py) excludes these
    // specifically until/unless that's added.
    std::vector<search::Action> getLegalActions(const BattleContext &bc) {
        std::vector<search::Action> actions;
        if (bc.outcome != Outcome::UNDECIDED) {
            return actions;
        }
        if (bc.inputState == InputState::CARD_SELECT) {
            return search::Action::enumerateCardSelectActions(bc);
        }
        // SCRY was missing entirely, so this returned an EMPTY vector for it and every
        // caller that assumes at least one legal action segfaulted -- nativeHeuristicPick's
        // `return legal[0]` (slaythespire.cpp:1277) during a rollout, reached via
        // nativeHeuristicPickFast's non-PLAYER_NORMAL fallback. The tree was equally blind:
        // nativeExpandLeaf/nativeRunMctsSearch assign node->actions from this function, so a
        // Scry node got zero edges and could not be searched at all.
        //
        // Each subset of the scry view is one legal "discard these" choice, mask-encoded --
        // the same enumeration BattleScumSearcher2.cpp:189 already used, kept identical so the
        // two searchers agree on what a Scry decision even is.
        if (bc.inputState == InputState::SCRY) {
            for (int mask = 0; mask < (1 << bc.scryCount); ++mask) {
                actions.emplace_back(search::ActionType::SCRY, mask);
            }
            return actions;
        }
        if (bc.inputState == InputState::PLAYER_NORMAL) {
            if (bc.isCardPlayAllowed()) {
                for (int handIdx = 0; handIdx < bc.cards.cardsInHand; ++handIdx) {
                    const auto &c = bc.cards.hand[handIdx];
                    if (!c.canUseOnAnyTarget(bc)) {
                        continue;
                    }
                    if (c.requiresTarget()) {
                        for (int tIdx = 0; tIdx < bc.monsters.monsterCount; ++tIdx) {
                            if (!bc.monsters.arr[tIdx].isTargetable()) {
                                continue;
                            }
                            actions.emplace_back(search::ActionType::CARD, handIdx, tIdx);
                        }
                    } else {
                        actions.emplace_back(search::ActionType::CARD, handIdx);
                    }
                }
            }
            for (int potionIdx = 0; potionIdx < bc.potionCapacity; ++potionIdx) {
                const auto p = bc.potions[potionIdx];
                if (p == Potion::INVALID || p == Potion::EMPTY_POTION_SLOT) {
                    continue;
                }
                // discard is always legal for any held potion, per isValidPotionAction
                actions.emplace_back(search::ActionType::POTION, potionIdx, 6);
                if (p == Potion::FAIRY_POTION) {
                    continue;  // not manually drinkable -- see isValidPotionAction
                }
                if (potionRequiresTarget(p)) {
                    for (int tIdx = 0; tIdx < bc.monsters.monsterCount; ++tIdx) {
                        if (!bc.monsters.arr[tIdx].isTargetable()) {
                            continue;
                        }
                        actions.emplace_back(search::ActionType::POTION, potionIdx, tIdx);
                    }
                } else {
                    actions.emplace_back(search::ActionType::POTION, potionIdx, 0);
                }
            }
            actions.emplace_back(search::ActionType::END_TURN);
        }
        return actions;
    }

    std::pair<int,int> getMonsterMoveDamage(const BattleContext &bc, int monsterIdx) {
        if (monsterIdx < 0 || monsterIdx >= bc.monsters.monsterCount) {
            return {0, 0};
        }
        const auto info = bc.monsters.arr[monsterIdx].getMoveBaseDamage(bc);
        return {info.damage, info.attackCount};
    }

    int getPlayerStatusValue(const BattleContext &bc, const std::string &name) {
        constexpr int n = sizeof(playerStatusEnumStrings) / sizeof(char*);
        for (int i = 0; i < n; ++i) {
            if (name == playerStatusEnumStrings[i]) {
                return bc.player.getStatusRuntime(static_cast<PlayerStatus>(i));
            }
        }
        throw std::runtime_error("getPlayerStatusValue: unknown PlayerStatus name: " + name);
    }

    int getMonsterStatusValue(const BattleContext &bc, int monsterIdx, const std::string &name) {
        if (monsterIdx < 0 || monsterIdx >= bc.monsters.monsterCount) {
            return 0;
        }
        const Monster &m = bc.monsters.arr[monsterIdx];
        // getStatus<MS::X>() already returns 0 when the status isn't
        // present, so this is safe for any monster regardless of what it
        // actually has. Explicit name->template mapping (see header note on
        // why not an ordinal index).
        if (name == "TIME_WARP")     return m.getStatus<MS::TIME_WARP>();
        if (name == "POISON")        return m.getStatus<MS::POISON>();
        if (name == "PLATED_ARMOR")  return m.getStatus<MS::PLATED_ARMOR>();
        if (name == "ARTIFACT")      return m.getStatus<MS::ARTIFACT>();
        if (name == "METALLICIZE")   return m.getStatus<MS::METALLICIZE>();
        if (name == "MODE_SHIFT")    return m.getStatus<MS::MODE_SHIFT>();
        if (name == "INTANGIBLE")    return m.getStatus<MS::INTANGIBLE>();
        if (name == "CURL_UP")       return m.getStatus<MS::CURL_UP>();
        throw std::runtime_error("getMonsterStatusValue: unknown/un-wired MonsterStatus name: " + name);
    }

    int getMonsterMiscInfo(const BattleContext &bc, int monsterIdx) {
        // Monster::miscInfo is a raw per-monster-type-specific int (stab
        // count, charge counters, etc. -- NOT part of the MS:: status enum,
        // so it can't go through getMonsterStatusValue's getStatus<MS::X>()
        // dispatch above). For Time Eater specifically, MonsterSpecific.cpp's
        // own move-selection logic reads it as a bool "have I already used
        // Haste this fight" flag (see the `const bool usedHaste = miscInfo;`
        // check gating TIME_EATER_HASTE) -- exposed as a raw passthrough
        // here rather than a Time-Eater-specific bool getter, since the
        // interpretation genuinely varies by monster type; the Python side
        // is expected to gate its use by monster identity (m.name), same as
        // monsters.py already keys embeddings off Monster::name.
        if (monsterIdx < 0 || monsterIdx >= bc.monsters.monsterCount) {
            return 0;
        }
        return bc.monsters.arr[monsterIdx].miscInfo;
    }

    MoveCategory classifyMonsterMove(const BattleContext &bc, int monsterIdx) {
        MoveCategory info;
        if (monsterIdx < 0 || monsterIdx >= bc.monsters.monsterCount) {
            return info;
        }
        if (bc.monsters.arr[monsterIdx].isDeadOrEscaped()) {
            return info;
        }

        // Pure read: BattleContext(bc) is a full value copy (RNG state
        // included), so running the move here never advances or otherwise
        // touches the real bc's own RNG stream or game state.
        BattleContext sample(bc);
        Monster &m = sample.monsters.arr[monsterIdx];

        const int selfStrBefore = m.strength;
        const int selfArtifactBefore = m.getStatus<MS::ARTIFACT>();
        const int selfPlatedBefore = m.getStatus<MS::PLATED_ARMOR>();
        const int selfMetalBefore = m.getStatus<MS::METALLICIZE>();
        std::vector<int> allyStrBefore;
        for (int i = 0; i < sample.monsters.monsterCount; ++i) {
            allyStrBefore.push_back(sample.monsters.arr[i].strength);
        }
        const int weakBefore = sample.player.getStatusRuntime(PS::WEAK);
        const int vulnBefore = sample.player.getStatusRuntime(PS::VULNERABLE);
        const int frailBefore = sample.player.getStatusRuntime(PS::FRAIL);
        const int strBefore = sample.player.strength;
        const int dexBefore = sample.player.dexterity;

        m.takeTurn(sample);
        sample.executeActions();

        if (m.strength > selfStrBefore
                || m.getStatus<MS::ARTIFACT>() > selfArtifactBefore
                || m.getStatus<MS::PLATED_ARMOR>() > selfPlatedBefore
                || m.getStatus<MS::METALLICIZE>() > selfMetalBefore) {
            info.self_buffs = true;
        }
        for (int i = 0; i < sample.monsters.monsterCount; ++i) {
            if (i != monsterIdx && sample.monsters.arr[i].strength > allyStrBefore[i]) {
                info.buffs_ally = true;
                break;
            }
        }
        if (sample.player.getStatusRuntime(PS::WEAK) > weakBefore
                || sample.player.getStatusRuntime(PS::VULNERABLE) > vulnBefore
                || sample.player.getStatusRuntime(PS::FRAIL) > frailBefore
                || sample.player.strength < strBefore
                || sample.player.dexterity < dexBefore) {
            info.debuffs_player = true;
        }
        return info;
    }

}
