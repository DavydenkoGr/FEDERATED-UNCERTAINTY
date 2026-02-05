import os
from typing import Final

# general paths
DATASETS_PATH: Final[str] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "datasets")
)
RESULTS_PATH: Final[str] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)
RESULTS_PATH_AL: Final[str] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results_al")
)
RESULTS_PATH_AL_RND: Final[str] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results_al_rnd")
)
PLOTS_PATH: Final[str] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "plots")
)

# dataset paths
UCI_PATH: Final[str] = os.path.join(DATASETS_PATH, "UCI")
SINE_PATH: Final[str] = os.path.join(DATASETS_PATH, "SINE")
DOTS_PATH: Final[str] = os.path.join(DATASETS_PATH, "DOTS")
ARROW_PATH: Final[str] = os.path.join(DATASETS_PATH, "ARROW")
ACS_PATH: Final[str] = os.path.join(DATASETS_PATH, "ACS")
CITYSCAPES_PATH: Final[str] = os.path.join(DATASETS_PATH, "CITYSCAPES")
