from enum import Enum


class UncertaintyType(Enum):
    RISK = "Risk"
    MAHALANOBIS = "Mahalanobis"
    GMM = "GMM"


class OTTarget(Enum):
    BALL = "Ball"
    EXP = "Exp"
    BETA = "Beta"


class SamplingMethod(Enum):
    RANDOM = "Random"
    SOBOL = "Sobol"
    GRID = "Grid"


class ScalingType(Enum):
    GLOBAL = "Global"
    FEATURE_WISE = "FeatureWise"
