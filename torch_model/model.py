import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import pdb


def findConv2dOutShape(hin, win, conv, pool=2):
    # get conv arguments
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    hout = np.floor(
        (hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    wout = np.floor(
        (win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )

    if pool:
        hout /= pool
        wout /= pool
    return int(hout), int(wout)


class SimpleModel(nn.Module):
    def __init__(self, input_shape: list, initial_filters=8, num_classes=36):
        super(SimpleModel, self).__init__()
        self.input_shape = input_shape

        self.w = self.input_shape[0]
        self.h = self.input_shape[1]
        self.ch = self.input_shape[2]
        self.init_f = initial_filters
        self.dropout_rate = 0.25

        self.build_model()

    def build_model(self):

        # Convolution Layers
        self.conv1 = nn.Conv2d(self.ch, self.init_f, kernel_size=3)
        h, w = findConv2dOutShape(self.h, self.w, self.conv1)
        self.conv2 = nn.Conv2d(self.init_f, 2 * self.init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv2)
        self.conv3 = nn.Conv2d(2 * self.init_f, 4 * self.init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv3)
        self.conv4 = nn.Conv2d(4 * self.init_f, 8 * self.init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv4)

        # compute the flatten size
        self.num_flatten = h * w * 8 * self.init_f
        self.fc1 = nn.Linear(self.num_flatten, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, X):

        # Convolution & Pool Layers
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, self.num_flatten)

        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)

        return X


class MobileNet(nn.Module):
    def __init__(self, input_shape: list, num_classes=36):
        super(MobileNet, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.build_model()

    def build_model(self):

        self.model = torchvision.models.mobilenet_v2(pretrained=True)

        fully_conected_layers = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )

        self.model.classifier = fully_conected_layers

        return

    def forward(self, X):

        X = self.model(X)
        return X

class ResNet50(nn.Module):
    def __init__(self, input_shape: list, num_classes=36):
        super(ResNet50, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.build_model()

    def build_model(self):

        self.model = torchvision.models.resnet50(pretrained=True)

        fully_conected_layers = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )

        self.model.fc = fully_conected_layers

        return

    def forward(self, X):

        X = self.model(X)
        return X
