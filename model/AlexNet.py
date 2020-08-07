import torch
import torch.nn as nn
from collections import OrderedDict

# parameters
n_classes = 10


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layerC1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6,
                      kernel_size=(5, 5), stride=1),
            nn.Tanh()
        )

        self.layerS2 = nn.AvgPool2d(kernel_size=2)

        self.layerC3 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16,
                      kernel_size=(5, 5), stride=1),
            nn.Tanh()
        )

        self.layerS4 = nn.AvgPool2d(kernel_size=2)

        self.layerC5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120,
                      kernel_size=(5, 5), stride=1),
            nn.Tanh()
        )
        #F6 and outputlayer are classifiers
        self.layerF6 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh()
        )

        self.outputLayer = nn.Sequential(
            nn.Linear(in_features=84, out_features=n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layerC1(x)
        # print(x.size())
        x = self.layerS2(x)
        # print(x.size())
        x = self.layerC3(x)
        # print(x.size())
        x = self.layerS4(x)
        # print(x.size())
        x = self.layerC5(x)
        # print(x.size())
        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.layerF6(x)
        # print(x.size())
        x = self.outputLayer(x)

        return x
__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = 'model_list/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model