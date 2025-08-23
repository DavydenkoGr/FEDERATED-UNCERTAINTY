from enum import Enum


class UncertaintyType(Enum):
    RISK = "Risk"
    MAHALANOBIS = "Mahalanobis"
    GMM = "GMM"


class VectorQuantileModel(Enum):
    OTCP = "OTCP"
    CPFLOW = "CPFlow"
    ENTROPIC_OT = "EntropicOT"
