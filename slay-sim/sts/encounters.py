"""Named monster groups, organized by act and tier, so demos/tests/search
benchmarks share one source of truth for "what fight is this".

Act 1 basic/elite/boss and a first slice of Act 2 are covered; the rest of
Act 2 and all of Acts 3-4 aren't implemented yet (same pattern, just more
monster classes to write in enemies.py -- not an architecture change) --
see the project notes for what's still open."""

from __future__ import annotations

import random
from typing import List, Optional

from .enemies import (
    Monster, JawWorm, Cultist, Louse, AcidSlimeM, SpikeSlimeM, BlueSlaver,
    Looter, GremlinNob, Lagavulin, Guardian, Sentry, MadGremlin,
    SneakyGremlin, FatGremlin, Byrd, Mystic, Centurion, Champ,
    Automaton, Collector, GremlinWizard, GremlinLeader,
    Darkling, OrbWalker, WrithingMass, Spiker, Repulsor, Exploder,
    SphericGuardian, Nemesis, Dagger, Reptomancer, AwakenedOneCultist,
    AwakenedOne, TimeEater, DonuDeca,
)


def encounter_jaw_worm() -> List[Monster]:
    return [JawWorm()]


def encounter_cultist() -> List[Monster]:
    return [Cultist()]


def encounter_louse_pair(rng: Optional[random.Random] = None) -> List[Monster]:
    rng = rng or random.Random()
    return [Louse(rng), Louse(rng)]


def encounter_acid_slime() -> List[Monster]:
    return [AcidSlimeM()]


def encounter_spike_slime() -> List[Monster]:
    return [SpikeSlimeM()]


def encounter_slaver_and_looter() -> List[Monster]:
    return [BlueSlaver(), Looter()]


def encounter_gremlin_gang() -> List[Monster]:
    """4-monster swarm: real Act 1 "Gremlin Gang" fight. All 4 sub-monsters
    have deterministic single-move AI (no chance branching at all), so this
    stresses the action-space axis -- many monsters, many single-target
    choices -- rather than chance-node branching."""
    return [MadGremlin(), MadGremlin(), SneakyGremlin(), FatGremlin()]


def encounter_gremlin_nob() -> List[Monster]:
    return [GremlinNob()]


def encounter_lagavulin() -> List[Monster]:
    return [Lagavulin()]


def encounter_sentries() -> List[Monster]:
    """3-monster elite fight, deterministic AI throughout (see Sentry)."""
    return [Sentry(True), Sentry(False), Sentry(True)]


def encounter_guardian() -> List[Monster]:
    return [Guardian()]


def encounter_byrd() -> List[Monster]:
    return [Byrd()]


def encounter_centurion_mystic() -> List[Monster]:
    """Act 2 elite pair -- the roster's cross-monster-interaction case
    (Mystic buffs/shields Centurion instead of acting on the player)."""
    return [Centurion(), Mystic()]


def encounter_champ() -> List[Monster]:
    return [Champ()]


def encounter_automaton() -> List[Monster]:
    """Act 2 boss: spawns 2 Bronze Orbs on its opening turn."""
    return [Automaton()]


def encounter_collector() -> List[Monster]:
    """Act 2 boss: spawns 2 Torch Heads on its opening turn."""
    return [Collector()]


def encounter_gremlin_leader() -> List[Monster]:
    """Act 2 elite: starts alongside a Gremlin Wizard + Fat Gremlin escort
    that Gremlin Leader re-summons (Rally) if either dies."""
    return [GremlinWizard(), FatGremlin(), GremlinLeader()]


def encounter_three_darklings() -> List[Monster]:
    return [Darkling(55), Darkling(58), Darkling(51)]  # A20


def encounter_orb_walker() -> List[Monster]:
    return [OrbWalker()]


def encounter_writhing_mass() -> List[Monster]:
    return [WrithingMass()]


def encounter_three_shapes() -> List[Monster]:
    return [Spiker(), Repulsor(34), Repulsor(33)]


def encounter_sphere_and_two_shapes() -> List[Monster]:
    return [Exploder(), Repulsor(34), SphericGuardian()]


def encounter_four_shapes() -> List[Monster]:
    return [Spiker(), Repulsor(34), Repulsor(33), Exploder()]


def encounter_nemesis() -> List[Monster]:
    return [Nemesis()]


def encounter_reptomancer() -> List[Monster]:
    """Act 3 elite: starts with 2 Daggers already alive; Reptomancer
    re-summons (Preparation) whenever fewer than 2 are alive."""
    return [Dagger(25), Reptomancer(), Dagger(20)]


def encounter_spheric_guardian() -> List[Monster]:
    return [SphericGuardian()]


def encounter_awakened_one() -> List[Monster]:
    return [AwakenedOneCultist(), AwakenedOneCultist(), AwakenedOne()]


def encounter_time_eater() -> List[Monster]:
    return [TimeEater()]


def encounter_donu_and_deca() -> List[Monster]:
    return [DonuDeca(starts_attacking=True), DonuDeca(starts_attacking=False)]


ACT1_BASIC = [
    encounter_jaw_worm, encounter_cultist, encounter_louse_pair,
    encounter_acid_slime, encounter_spike_slime, encounter_slaver_and_looter,
    encounter_gremlin_gang,
]
ACT1_ELITE = [encounter_gremlin_nob, encounter_lagavulin, encounter_sentries]
ACT1_BOSS = [encounter_guardian]

ACT2_BASIC = [encounter_byrd]
ACT2_ELITE = [encounter_centurion_mystic, encounter_gremlin_leader]
ACT2_BOSS = [encounter_champ, encounter_automaton, encounter_collector]

ACT3_BASIC = [
    encounter_three_darklings, encounter_orb_walker, encounter_writhing_mass,
    encounter_three_shapes, encounter_sphere_and_two_shapes, encounter_four_shapes,
]
ACT3_ELITE = [encounter_nemesis, encounter_reptomancer, encounter_spheric_guardian]
ACT3_BOSS = [encounter_awakened_one, encounter_time_eater, encounter_donu_and_deca]
