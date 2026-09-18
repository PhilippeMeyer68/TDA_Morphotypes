"""Settings of the paper (de Rose, Meyer & Bertrand, Algorithms 2023).

The published results come from two sets of notebooks whose preprocessing
differs slightly: sections 3 and 5 (anomalies, morphotypes) on one side and
section 4 (gender discrimination index) on the other. Both are encoded here so
that every table and figure is reproduced exactly; see
``docs/reproducibility.md`` for how each value was recovered.
"""

from __future__ import annotations

from dataclasses import dataclass

SEXES = ("male", "female")
RANDOM_STATE = 42


@dataclass(frozen=True)
class DiagramSetting:
    """How a scan is turned into persistence diagrams (alpha complex, H0-H2).

    Filtration values of the alpha complex are squared radii, expressed in the
    squared unit of the normalised point cloud.
    """

    name: str
    target_height: float
    height_decimals: int | None
    min_persistence: float
    trunk: bool = False


# Sections 3 and 5: bodies scaled to 1.70 m, the measured height being rounded
# to the centimetre; features of persistence below 3 cm^2 are discarded.
BODY_M = DiagramSetting("body_m", target_height=1.7, height_decimals=2, min_persistence=3e-4)

# Section 4: bodies scaled to 170 cm (unrounded height).
BODY_CM = DiagramSetting("body_cm", target_height=170.0, height_decimals=None, min_persistence=3.0)

# Section 4.2: trunks isolated from bodies scaled to 170 cm.
TRUNK = DiagramSetting(
    "trunk", target_height=170.0, height_decimals=None, min_persistence=5500 / 999, trunk=True
)

# Section 4: the body silhouettes use the BODY_CM diagrams with a higher threshold.
BODY_SILHOUETTE_MIN_PERSISTENCE = 5.25


@dataclass(frozen=True)
class SilhouetteSetting:
    """Vectorisation of diagrams by persistence silhouettes, one block per degree.

    ``weights`` names a weight function per degree (see ``silhouettes.WEIGHTS``).
    Degrees listed in ``zero_degrees`` contribute a block of zeros.
    """

    resolutions: tuple[int, int, int] = (25, 250, 250)
    weights: tuple[str, str, str] = ("uniform", "birth", "birth")
    zero_degrees: tuple[int, ...] = ()


# Sections 2.2 and 5: 25 + 250 + 250 samples, H1/H2 weighted by birth time.
MORPHOTYPE_SILHOUETTE = SilhouetteSetting()

# Section 4.1 (Figure 14, Table 2 "Body"). In the original run every H1
# silhouette was identically zero: some empty H1 diagrams were stored as 1-D
# arrays, which made gudhi <= 3.7 fail to fit the H1 sampling range silently.
# Setting zero_degrees=() gives the corrected vectors (mean GDIs change by less
# than 0.001 for Ward).
GDI_BODY_SILHOUETTE = SilhouetteSetting(zero_degrees=(1,))

# Section 4.2 (Figure 17, Table 2 "Trunk").
GDI_TRUNK_SILHOUETTE = SilhouetteSetting(weights=("ratio", "ratio", "ratio"))

# (2, 2)-Wasserstein distance; the distance between two scans combines the
# per-degree distances in the l2 sense. In the section 3 computation the H0 term
# came out as zero for every pair (an artefact of the gudhi/POT versions used at
# the time, triggered by the essential class (0, inf) present in every H0
# diagram), so the published anomaly detection effectively uses H1 and H2 only.
WASSERSTEIN_ORDER = 2.0
WASSERSTEIN_INTERNAL_P = 2.0
ANOMALY_DEGREES = (1, 2)
GDI_DEGREES = (0, 1, 2)

# Scan anomalies found by visual inspection (section 3).
KNOWN_ANOMALIES = {
    "male": ("SPRING2277", "SPRING2882", "SPRING2921", "SPRING2962", "SPRING4624"),
    "female": ("SPRING1212", "SPRING2340", "SPRING2825", "SPRING2997"),
}

# Number of clusters explored for the GDI curves and the quality indices.
GDI_N_CLUSTERS = range(2, 31)
INDEX_N_CLUSTERS = range(2, 30)
INDEX_OPTIMA_WINDOW = (3, 20)  # where local optima of the indices are looked for

# Truncations used in the paper.
ANOMALY_DENDROGRAM_LEAVES = {"male": 21, "female": 23}
MORPHOTYPE_N_CLUSTERS = {"male": 8, "female": 7}
MORPHOTYPE_DENDROGRAM_LEAVES = 20

# Individuals shown in the illustrative figures.
EXAMPLE_SCAN = ("male", "SPRING0013")
ANOMALY_EXAMPLE_SCAN = ("male", "SPRING2962")
ANOMALY_EXAMPLE_H2 = (3, 5, 6)  # right leg, left leg, principal H2 (Figures 9 and 10)
NORMALISATION_EXAMPLE = (("male", "SPRING0105"), ("male", "SPRING0071"), ("male", "SPRING0207"))
