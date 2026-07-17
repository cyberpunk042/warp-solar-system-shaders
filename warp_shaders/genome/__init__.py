"""genome — engine library: biological-compression processes that TRANSFORM the real board.

Each process conserves matter — it uses what it transforms, it never spawns. Process 1: tokenization
(the board becomes a cloud of tokens). Process 2: base-pair bounding (tokens bind in twos).
"""

from .tokenize import tokenize_card, TokenCloud  # noqa: F401
from .basepair import bind_pairs, BasePairs  # noqa: F401
from .helix import wind_helix, DoubleHelix  # noqa: F401
from .nucleosome import wrap_nucleosomes, Nucleosomes  # noqa: F401
from .fiber import coil_fiber, Fiber  # noqa: F401
from .chromosome import fold_chromosome, Chromosome  # noqa: F401
from .journey import genome_journey, STAGE_NAMES  # noqa: F401
