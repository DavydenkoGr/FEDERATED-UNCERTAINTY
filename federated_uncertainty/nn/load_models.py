import torch.nn
from .constants import ModelName
from .resnet import ResNet18
from .vgg import VGG
from .simple_models import ShallowNet, LinearModel


def get_model(model_name: str, n_classes: int = 10, **kwargs) -> torch.nn.Module:
    match ModelName(model_name):
        case ModelName.RESNET18:
            return ResNet18(n_classes=n_classes)
        case ModelName.VGG11:
            return VGG(vgg_name="VGG11", n_classes=n_classes)
        case ModelName.VGG13:
            return VGG(vgg_name="VGG13", n_classes=n_classes)
        case ModelName.VGG16:
            return VGG(vgg_name="VGG16", n_classes=n_classes)
        case ModelName.VGG19:
            return VGG(vgg_name="VGG19", n_classes=n_classes)
        case ModelName.SHALLOWNET:
            return ShallowNet(n_classes=n_classes, **kwargs)
        case ModelName.LINEAR_MODEL:
            return LinearModel(n_classes=n_classes, **kwargs)
        case _:
            raise ValueError(
                f"{model_name} --  no such neural network is available. ",
                f"Available options are: {[element.value for element in ModelName]}",
            )
