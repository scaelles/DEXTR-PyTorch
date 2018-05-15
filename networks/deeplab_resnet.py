import torch.nn as nn
import torchvision.models.resnet as resnet
import torch
import numpy as np
from copy import deepcopy
import os
from torch.nn import functional as F
from mypath import Path

affine_par = True


def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation=dilation_)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ClassifierModule(nn.Module):

    def __init__(self, dilation_series, padding_series, n_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, n_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing module
    """
    def __init__(self, in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=1):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage_1(in_features, size) for size in sizes])
        self.bottleneck = self._make_stage_2(in_features * (len(sizes)//4 + 1), out_features)
        self.relu = nn.ReLU()
        self.final = nn.Conv2d(out_features, n_classes, kernel_size=1)

    def _make_stage_1(self, in_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_features, in_features//4, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(in_features//4, affine=affine_par)
        relu = nn.ReLU(inplace=True)

        return nn.Sequential(prior, conv, bn, relu)

    def _make_stage_2(self, in_features, out_features):
        conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features, affine=affine_par)
        relu = nn.ReLU(inplace=True)

        return nn.Sequential(conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages]
        priors.append(feats)
        bottle = self.relu(self.bottleneck(torch.cat(priors, 1)))
        out = self.final(bottle)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, n_classes, nInputChannels=3, classifier="atrous",
                 dilations=(2, 4), strides=(2, 2, 2, 1, 1), _print=False):
        if _print:
            print("Constructing ResNet model...")
            print("Dilations: {}".format(dilations))
            print("Number of classes: {}".format(n_classes))
            print("Number of Input Channels: {}".format(nInputChannels))
        self.inplanes = 64
        self.classifier = classifier
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3], dilation__=dilations[0])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4], dilation__=dilations[1])

        if classifier == "atrous":
            if _print:
                print('Initializing classifier: A-trous pyramid')
            self.layer5 = self._make_pred_layer(ClassifierModule, [6, 12, 18, 24], [6, 12, 18, 24], n_classes=n_classes)
        elif classifier == "psp":
            if _print:
                print('Initializing classifier: PSP')
            self.layer5 = PSPModule(in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=n_classes)
        else:
            self.layer5 = None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = [block(self.inplanes, planes, stride, dilation_=dilation__, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, n_classes):
        return block(dilation_series, padding_series, n_classes)

    def forward(self, x, bbox=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.layer5 is not None:
            x = self.layer5(x)
        return x

    def load_pretrained_ms(self, base_network, nInputChannels=3):
        flag = 0
        for module, module_ori in zip(self.modules(), base_network.Scale.modules()):
            if isinstance(module, nn.Conv2d) and isinstance(module_ori, nn.Conv2d):
                if not flag and nInputChannels != 3:
                    module.weight[:, :3, :, :].data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                    for i in range(3, int(module.weight.data.shape[1])):
                        module.weight[:, i, :, :].data = deepcopy(module_ori.weight[:, -1, :, :][:, np.newaxis, :, :].data)
                    flag = 1
                elif module.weight.data.shape == module_ori.weight.data.shape:
                    module.weight.data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                else:
                    print('Skipping Conv layer with size: {} and target size: {}'
                          .format(module.weight.data.shape, module_ori.weight.data.shape))
            elif isinstance(module, nn.BatchNorm2d) and isinstance(module_ori, nn.BatchNorm2d) \
                    and module.weight.data.shape == module_ori.weight.data.shape:
                module.weight.data = deepcopy(module_ori.weight.data)
                module.bias.data = deepcopy(module_ori.bias.data)


def resnet101(n_classes, pretrained=False, nInputChannels=3, classifier="atrous",
              dilations=(2, 4), strides=(2, 2, 2, 1, 1)):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], n_classes, nInputChannels=nInputChannels,
                   classifier=classifier, dilations=dilations, strides=strides, _print=True)
    if pretrained:
        model_full = Res_Deeplab(n_classes, pretrained=pretrained)
        model.load_pretrained_ms(model_full, nInputChannels=nInputChannels)
    return model


class MS_Deeplab(nn.Module):
    def __init__(self, block, NoLabels, nInputChannels=3):
        super(MS_Deeplab, self).__init__()
        self.Scale = ResNet(block, [3, 4, 23, 3], NoLabels, nInputChannels=nInputChannels)

    def forward(self, x):
        input_size = x.size()[2]
        self.interp1 = nn.Upsample(size=(int(input_size*0.75)+1, int(input_size*0.75)+1), mode='bilinear', align_corners=True)
        self.interp2 = nn.Upsample(size=(int(input_size*0.5)+1, int(input_size*0.5)+1), mode='bilinear', align_corners=True)
        self.interp3 = nn.Upsample(size=(outS(input_size), outS(input_size)), mode='bilinear', align_corners=True)
        out = []
        x2 = self.interp1(x)
        x3 = self.interp2(x)
        out.append(self.Scale(x))  # for original scale
        out.append(self.interp3(self.Scale(x2)))  # for 0.75x scale
        out.append(self.Scale(x3))  # for 0.5x scale

        x2Out_interp = out[1]
        x3Out_interp = self.interp3(out[2])
        temp1 = torch.max(out[0], x2Out_interp)
        out.append(torch.max(temp1, x3Out_interp))
        return out[-1]


def Res_Deeplab(n_classes=21, pretrained=False):
    model = MS_Deeplab(Bottleneck, n_classes)
    if pretrained:
        pth_model = 'MS_DeepLab_resnet_trained_VOC.pth'
        saved_state_dict = torch.load(os.path.join(Path.models_dir(), pth_model),
                                      map_location=lambda storage, loc: storage)
        if n_classes != 21:
            for i in saved_state_dict:
                i_parts = i.split('.')
                if i_parts[1] == 'layer5':
                    saved_state_dict[i] = model.state_dict()[i]
        model.load_state_dict(saved_state_dict)
    return model


def get_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4, model.layer5]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.layer5]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr*((1-float(iter_)/max_iter)**power)
