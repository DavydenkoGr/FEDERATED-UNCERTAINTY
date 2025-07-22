from enum import Enum


class UncertaintyType(Enum):
    RISK = "Risk"
    MAHALANOBIS = "Mahalanobis"


class VectorQuantileModel(Enum):
    OTCP = "OTCP"
    CPFLOW = "CPFlow"
