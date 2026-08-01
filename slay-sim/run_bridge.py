"""Entry point for CommunicationMod's `command=` config line.

CommunicationMod builds its subprocess command by naive whitespace-splitting
a single config string (`command.trim().split("\\s+")`), with no shell and no
working directory set on the ProcessBuilder (confirmed by decompiling
CommunicationMod.class) -- so `python -m sts.bridge.communication_mod` would
fail to find the `sts` package once launched from inside the game, whose
process cwd is wherever ModTheSpire/the game itself sets it, not slay-sim/.

This script sidesteps both problems: its own path has no spaces (required,
since the naive split can't handle quoting), and it adds its own directory to
sys.path from __file__ rather than relying on cwd, so it works no matter what
directory the game process happens to be in.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sts.bridge.communication_mod import main

if __name__ == "__main__":
    main()
