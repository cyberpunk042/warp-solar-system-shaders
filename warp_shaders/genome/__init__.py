"""genome — engine library: biological-compression processes that TRANSFORM the real board.

Each process conserves matter — it uses what it transforms, it never spawns, and it chains from the
previous process's actual output. Process 1: tokenization (the board becomes a cloud of tokens).
Process 2: base-pair bounding (tokens bind in twos). Process 3: the double helix (the base-pair field
winds into DNA). Processes 4-6 (nucleosomes, 30nm fibre, chromosome) are being rebuilt honestly, one at
a time, each chaining from the one before.
"""

from .tokenize import tokenize_card, TokenCloud  # noqa: F401
from .basepair import bind_pairs, BasePairs, ordered_field_sites  # noqa: F401
from .helix import wind_helix, DoubleHelix  # noqa: F401
