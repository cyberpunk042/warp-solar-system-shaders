"""genome — engine library: biological-compression processes that TRANSFORM the real board.

Each process conserves matter — it uses what it transforms, it never spawns (the one exception is the
explicit, shown DNA replication in Process 8), and it chains from the previous process's actual output.
Process 1: tokenization. Process 2: base-pair bounding. Process 3: the double helices. Process 4:
nucleosomes (beads on a string). Process 5: the 30nm fibre. Process 6: telomeres (the strand's two ends
curl into t-loop caps). Process 7: the chromosome (single chromatid). Process 8: replication into the
metaphase X. The full ladder — token to chromosome.
"""

from .tokenize import tokenize_card, TokenCloud  # noqa: F401
from .basepair import bind_pairs, BasePairs, ordered_field_sites  # noqa: F401
from .helix import wind_helix, wound_positions, DoubleHelix  # noqa: F401
from .nucleosome import wrap_nucleosomes, Nucleosomes  # noqa: F401
from .fibre import coil_fibre, Fibre  # noqa: F401
from .telomere import cap_telomeres, Telomeres  # noqa: F401
from .chromosome import fold_chromosome, Chromosome  # noqa: F401
from .replication import replicate_chromosome, Replication  # noqa: F401
