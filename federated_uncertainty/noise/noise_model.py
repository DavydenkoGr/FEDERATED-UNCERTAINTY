from enum import Enum
import torch
import copy
import torch.quantization as tq


class NoiseType(Enum):
    QUANT = "Quant"
    NOIZE_WEIGHTS = "NoiseWeights"


NOISE_CHOICES = [e.name.lower() for e in NoiseType]


def get_noisy_model(model, noise_type: NoiseType, device, noise_level=None):
    """
    If use quantization, model should have fuse_model() method
    """
    noisy_model = copy.deepcopy(model)

    if noise_type.value == "NoiseWeights":
        for p in noisy_model.parameters():
            if p.data is not None:
                p.data.add_(torch.randn_like(p) * noise_level)
    elif noise_type.value == "Quant":
        qconfig = tq.get_default_qat_qconfig('fbgemm')
        model.qconfig = qconfig

        noisy_model.eval()
        noisy_model.fuse_model()
        tq.prepare_qat(model, inplace=True)
    
    return noisy_model.to(device)