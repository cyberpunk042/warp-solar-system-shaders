"""genome — engine library: biological-compression processes that TRANSFORM the real board.

Each process conserves matter — it uses what it transforms, it never spawns, and it chains from the
previous process's actual output. Process 1: tokenization. Process 2: base-pair bounding. Process 3: the
double helices. Process 4: nucleosomes (beads on a string). Process 5: the 30nm fibre. Process 6:
telomeres (the strand's two ends curl into t-loop caps). Process 7: the chromatid fold (the capped fibre
winds onto a coil scaffold into the condensed chromosome arm). Run back-to-back on one timeline this
molecular ladder is the whole genome compression as one continuous animation (``scenes/warp_genome``).
"""

from .tokenize import tokenize_card, TokenCloud  # noqa: F401
from .basepair import bind_pairs, BasePairs, ordered_field_sites  # noqa: F401
from .helix import wind_helix, wound_positions, DoubleHelix  # noqa: F401
from .nucleosome import wrap_nucleosomes, Nucleosomes  # noqa: F401
from .fibre import coil_fibre, Fibre  # noqa: F401
from .telomere import cap_telomeres, Telomeres  # noqa: F401
from .chromatid import fold_chromatid, Chromatid  # noqa: F401
