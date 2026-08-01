//
// Created by gamerpuppy on 7/24/2021.
//

#ifndef STS_LIGHTSPEED_CARDMANAGER_H
#define STS_LIGHTSPEED_CARDMANAGER_H

#include <vector>
#include <array>

#include "sts_common.h"

#include "combat/CardInstance.h"
#include "game/Random.h"
#include "game/Deck.h"

namespace sts {

    class GameContext;
    class Card;

    struct CardManager {

        static constexpr int MAX_HAND_SIZE = 10;
        // 64 -- matches the original engine default. See sts_common.h's
        // sts_card_manager_use_fixed_list comment for the full investigation:
        // this value is currently moot (fixed_list is disabled, std::vector
        // used instead) since raising it did not resolve the underlying,
        // still-unexplained assert failures found there.
        static constexpr int MAX_GROUP_SIZE = 64;

        int nextUniqueCardId = 0; // unique card ids that are less than the masterDeckSize are non-temporary
        int masterDeckSize = 0;

        int cardsInHand = 0;
        std::array<CardInstance, MAX_HAND_SIZE> hand;
        std::array<CardInstance, MAX_HAND_SIZE> limbo; // used only for end of turn during discard, for retained cards
        std::array<CardInstance,2> stasisCards { CardId::INVALID, CardId::INVALID }; // for bronze automaton fight

#ifdef sts_card_manager_use_fixed_list
        fixed_list<CardInstance, MAX_GROUP_SIZE> drawPile;
        fixed_list<CardInstance, MAX_GROUP_SIZE> discardPile;
        fixed_list<CardInstance, MAX_GROUP_SIZE> exhaustPile;
#else
        std::vector<CardInstance> drawPile;
        std::vector<CardInstance> discardPile;
        std::vector<CardInstance> exhaustPile;
#endif
        int handNormalityCount = 0;
        int handPainCount = 0;
        int strikeCount = 0;
        int handBloodCardCount = 0;
        int drawPileBloodCardCount = 0;
        int discardPileBloodCardCount = 0;

        void init(const GameContext &gc, BattleContext &bc); // returns count of innate cards

        void createDeckCardInstanceInDrawPile(const Card &card, int deckIdx, int drawIdx);
        void createTempCardInDrawPile(int insertIdx, CardInstance c);
        void createTempCardInDiscard(CardInstance c);
        void createTempCardInHand(CardInstance c);

        void removeFromDrawPileAtIdx(int idx);
        CardInstance popFromDrawPile();

        void removeFromHandAtIdx(int idx); // this method is dangerous if used in the wrong place.
        // Signed, to match CardInstance::getUniqueId's own int16_t. An unsigned
        // parameter here converted a -1 (CardInstance's unset default) to 65535,
        // which could never compare equal to any hand card's id -- so the played
        // card was silently left in hand instead of being removed.
        void removeFromHandById(std::int16_t uniqueId); // can do more than one card if they have the same uniqueId, does this happen?
        void removeFromDiscard(int idx);
        void removeFromExhaustPile(int idx);

        void moveToHand(const CardInstance &c);
        void moveToExhaustPile(const CardInstance &c);

        void insertToDrawPile(int drawPileIdx, const CardInstance &c);
        void moveToDrawPileTop(const CardInstance &c);
        void shuffleIntoDrawPile(Random &cardRandomRng, const CardInstance &c);

        void moveToDiscardPile(const CardInstance &c);
        void moveDiscardPileIntoToDrawPile();

        // **************
        void notifyAddCardToCombat(const CardInstance &c);
        void notifyRemoveFromCombat(const CardInstance &c);

        void notifyAddToHand(const CardInstance &c);
        void notifyRemoveFromHand(const CardInstance &c);

        void notifyAddToDrawPile(const CardInstance &c);
        void notifyRemoveFromDrawPile(const CardInstance &c);

        void notifyAddToDiscardPile(const CardInstance &c);
        void notifyRemoveFromDiscardPile(const CardInstance &c);
        // **************

        void eraseAtIdxInHand(int idx); // does not call notifyRemoveFromHand
        int getRandomCardIdxInHand(Random &rng);
        void resetAttributesAtEndOfTurn();

        // special helpers
        void draw(BattleContext &bc, int amount);
        void onTookDamage(); // update blood for blood, masterful stab
        void findAndUpgradeSpecialData(std::int16_t uniqueId, int amount);
        void onBuffCorruption();
    };

    std::ostream &operator <<(std::ostream &os, const CardManager &c);

}

#endif //STS_LIGHTSPEED_CARDMANAGER_H
