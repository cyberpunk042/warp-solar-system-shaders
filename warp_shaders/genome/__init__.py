"""genome — engine library: biological-compression processes that TRANSFORM the real board.

Each process conserves matter — it uses what it transforms, it never spawns, and it chains from the
previous process's actual output. Process 1: tokenization (the board becomes a cloud of tokens).
Process 2: base-pair bounding (tokens bind in twos). Process 3: the double helices (the base-pair field
winds into many DNA double helices). Process 4: nucleosomes (the helices bead up into beads-on-a-string).
Process 5: the 30nm fibre (the beads coil into solenoid fibres). Process 6 (chromosome) is being rebuilt
honestly, chaining from the one before.
"""

from .tokenize import tokenize_card, TokenCloud  # noqa: F401
from .basepair import bind_pairs, BasePairs, ordered_field_sites  # noqa: F401
from .helix import wind_helix, wound_positions, DoubleHelix  # noqa: F401
from .nucleosome import wrap_nucleosomes, Nucleosomes  # noqa: F401
from .fibre import coil_fibre, Fibre  # noqa: F401
