"""RL against Ironclad fights, backed by the sts_lightspeed C++ engine
(https://github.com/gamerpuppy/sts_lightspeed, cloned at /home/alvin/sts_lightspeed)
rather than our own sts/ Python simulator.

This is a separate, parallel effort from sts/ -- not a replacement. sts/
remains the exact, interpretable, transposition-tabled expectimax + MCTS
solver, useful on its own and as a verification cross-check (its Jaw Worm
numbers exactly matched sts_lightspeed's during validation). This package
exists specifically to get the training throughput and full Ironclad card
coverage (75 cards including Rampage/Blood for Blood/Searing Blow/Juggernaut,
which sts/ explicitly can't support -- see sts/cards.py's docstring) needed
to train an agent that learns to play with the whole card pool, not one
fixed deck.

Requires the compiled `slaythespire` module on the Python path (symlinked
into this venv's site-packages from /home/alvin/sts_lightspeed/build/).
"""
