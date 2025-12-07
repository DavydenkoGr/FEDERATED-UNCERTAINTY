"""VGG11/13/16/19 in Pytorch."""

from typing import List

import torch
import torch.nn as nn

VGG_NAME_TO_CONFIGURATION_DICT = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(torch.nn.Module):
    def __init__(self, vgg_name, n_classes):
        super(VGG, self).__init__()
        if n_classes == 10:
            self.classifier = torch.nn.Linear(512, n_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, n_classes),
            )
        self.features = self._make_layers(
            configuration=VGG_NAME_TO_CONFIGURATION_DICT[vgg_name]
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, configuration: List):
        layers = []
        in_channels = 3
        for x in configuration:
            if x == "M":
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    torch.nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(x),
                    torch.nn.ReLU(inplace=True),
                ]
                in_channels = x
        if hasattr(self.classifier, "out_features"):
            layers += [torch.nn.AvgPool2d(kernel_size=1, stride=1)]
        return torch.nn.Sequential(*layers)


class QuantVGG(VGG):
    def __init__(self, vgg_name, n_classes):
        super(QuantVGG, self).__init__(vgg_name, n_classes)
        # QuantStub: float -> int8
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub: int8 -> float
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        
        out = self.dequant(out)
        return out

    def fuse_model(self):
        """
        Layers fusing (Conv + BN + ReLU; Linear + ReLU head) for correct quantization 
        """
        modules_to_fuse = []
        for i in range(len(self.features) - 2):
            if (isinstance(self.features[i], nn.Conv2d) and
                isinstance(self.features[i + 1], nn.BatchNorm2d) and
                isinstance(self.features[i + 2], nn.ReLU)):

                modules_to_fuse.append([str(i), str(i + 1), str(i + 2)])
        
        if modules_to_fuse:
            torch.quantization.fuse_modules(self.features, modules_to_fuse, inplace=True)

        if isinstance(self.classifier, nn.Sequential):
            modules_to_fuse_clf = []
            for i in range(len(self.classifier) - 1):
                if (isinstance(self.classifier[i], nn.Linear) and
                    isinstance(self.classifier[i + 1], nn.ReLU)):
                    modules_to_fuse_clf.append([str(i), str(i + 1)])
            if modules_to_fuse_clf:
                torch.quantization.fuse_modules(self.classifier, modules_to_fuse_clf, inplace=True)