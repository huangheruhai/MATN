#!/usr/bin/env python
#-*- coding: utf-8 -*-
import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck
from torch.nn import functional as F
num_classes = 751  # market1501 751 cuhk 767 duke702
width_ratio=1.0
height_ratio=0.2

class BatchDrop(nn.Module):
    def __init__(self,h_ratio,w_ratio):
        super(BatchDrop,self).__init__()
        self.h_ratio=h_ratio
        self.w_ratio=w_ratio

    def forward(self,x):
        if self.training:
            h,w=x.size()[-2:]
            rh=int(self.h_ratio*h)
            rw=int(self.w_ratio*w)
            sx=random.randint(0,h-rh)
            xy=ramndom.randint(0,w-rw)
            mask=x.new_ones(x.size())
            mask[:,:,sx:sx+rh,sy:sy+rw]=0
            x=x*mask

        return x

def weights_init_kaiming(m):
    classname=m.__class__.__name__
    if classname.find('Linear')!=-1:
        nn.init.kaiming_normal_(m.weight,a=0,mode='fan_out')
        nn.init.constant_(m.bias,0.0)
    elif classname.find('Conv')!=-1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight,1.0,0.02)
            nn.init.constant_(m.bias,0.0)

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
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

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(torch.load('se_resnet50-ce0d4300.pth'))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']





def se_resnet50(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


pretrained_settings = {

    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    }

}

class ConvBlock(nn.Module):
    """Basic convolutional block.

    convolution + batch normalization + relu.
    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionA, self).__init__()
        mid_channels = out_channels // 4

        self.stream1 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
        )
        self.stream2 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
        )
        self.stream3 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
        )
        self.stream4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, mid_channels, 1),
        )

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        y = torch.cat([s1, s2, s3, s4], dim=1)
        return y


class InceptionB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionB, self).__init__()
        mid_channels = out_channels // 4

        self.stream1 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, s=2, p=1),
        )
        self.stream2 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
            ConvBlock(mid_channels, mid_channels, 3, s=2, p=1),
        )
        self.stream3 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBlock(in_channels, mid_channels * 2, 1),
        )

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        y = torch.cat([s1, s2, s3], dim=1)
        return y


class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""

    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # bilinear resizing
        x = F.upsample(x, (x.size(2) * 2, x.size(3) * 2), mode='bilinear', align_corners=True)
        # scaling conv
        x = self.conv2(x)
        return x


class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""

    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)

    Aim: Spatial Attention + Channel Attention

    Output: attention maps with shape identical to input.
    """

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        return y


class HardAttn(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""

    def __init__(self, in_channels):
        super(HardAttn, self).__init__()
        self.fc = nn.Linear(in_channels, 4 * 2)
        self.init_params()

    def init_params(self):
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([0, -0.75, 0, -0.25, 0, 0.25, 0, 0.75], dtype=torch.float))

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        # predict transformation parameters
        theta = torch.tanh(self.fc(x))
        theta = theta.view(-1, 4, 2)
        return theta


class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""

    def __init__(self, in_channels):
        super(HarmAttn, self).__init__()
        self.soft_attn = SoftAttn(in_channels)
        self.hard_attn = HardAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        theta = self.hard_attn(x)
        return y_soft_attn, theta


class HACNN(nn.Module):
    """Harmonious Attention Convolutional Neural Network.
    Reference:
        Li et al. Harmonious Attention Network for Person Re-identification. CVPR 2018.
    Public keys:
        - ``hacnn``: HACNN.
    """

    # Args:
    #    num_classes (int): number of classes to predict
    #    nchannels (list): number of channels AFTER concatenation
    #    feat_dim (int): feature dimension for a single stream
    #    learn_region (bool): whether to learn region features (i.e. local branch)

    def __init__(self, num_classes, loss='softmax', nchannels=[128, 256, 384], feat_dim=512, learn_region=True,
                 use_gpu=True, **kwargs):
        super(HACNN, self).__init__()
        self.loss = loss
        self.learn_region = learn_region
        self.use_gpu = use_gpu

        self.conv = ConvBlock(3, 32, 3, s=2, p=1)

        # Construct Inception + HarmAttn blocks
        # ============== Block 1 ==============
        self.inception1 = nn.Sequential(
            InceptionA(32, nchannels[0]),
            InceptionB(nchannels[0], nchannels[0]),
        )
        self.ha1 = HarmAttn(nchannels[0])

        # ============== Block 2 ==============
        self.inception2 = nn.Sequential(
            InceptionA(nchannels[0], nchannels[1]),
            InceptionB(nchannels[1], nchannels[1]),
        )
        self.ha2 = HarmAttn(nchannels[1])

        # ============== Block 3 ==============
        self.inception3 = nn.Sequential(
            InceptionA(nchannels[1], nchannels[2]),
            InceptionB(nchannels[2], nchannels[2]),
        )
        self.ha3 = HarmAttn(nchannels[2])

        self.fc_global = nn.Sequential(
            nn.Linear(nchannels[2], feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )
        self.classifier_global = nn.Linear(feat_dim, num_classes)

        if self.learn_region:
            self.init_scale_factors()
            self.local_conv1 = InceptionB(32, nchannels[0])
            self.local_conv2 = InceptionB(nchannels[0], nchannels[1])
            self.local_conv3 = InceptionB(nchannels[1], nchannels[2])
            self.fc_local0 = nn.Sequential(
                nn.Linear(nchannels[2] * 1, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(),
            )
            self.fc_local1 = nn.Sequential(
                nn.Linear(nchannels[2] * 1, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(),
            )
            self.fc_local2 = nn.Sequential(
                nn.Linear(nchannels[2] * 1, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(),
            )
            self.fc_local3 = nn.Sequential(
                nn.Linear(nchannels[2] * 1, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(),
            )
            # self.classifier_local = nn.Linear(feat_dim, num_classes)
            # self.feat_dim = feat_dim * 2
            self.classifier_local0 = nn.Linear(feat_dim, num_classes)
            self.classifier_local1 = nn.Linear(feat_dim, num_classes)
            self.classifier_local2 = nn.Linear(feat_dim, num_classes)
            self.classifier_local3 = nn.Linear(feat_dim, num_classes)
            self.classifier_local0.apply(weights_init_kaiming)
            self.classifier_local1.apply(weights_init_kaiming)
            self.classifier_local2.apply(weights_init_kaiming)
            self.classifier_local3.apply(weights_init_kaiming)




        else:
            self.feat_dim = feat_dim

    def init_scale_factors(self):
        # initialize scale factors (s_w, s_h) for four regions
        self.scale_factors = []
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))

    def stn(self, x, theta):
        """Performs spatial transform

        x: (batch, channel, height, width)
        theta: (batch, 2, 3)
        """
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def transform_theta(self, theta_i, region_idx):
        """Transforms theta to include (s_w, s_h), resulting in (batch, 2, 3)"""
        scale_factors = self.scale_factors[region_idx]
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:, :, :2] = scale_factors
        theta[:, :, -1] = theta_i
        if self.use_gpu: theta = theta.cuda()
        return theta

    def forward(self, x):
        assert x.size(2) == 160 and x.size(3) == 64, \
            'Input size does not match, expected (160, 64) but got ({}, {})'.format(x.size(2), x.size(3))
        x = self.conv(x)

        # ============== Block 1 ==============
        # global branch
        x1 = self.inception1(x)
        x1_attn, x1_theta = self.ha1(x1)
        x1_out = x1 * x1_attn
        # local branch
        if self.learn_region:
            x1_local_list = []
            for region_idx in range(4):
                x1_theta_i = x1_theta[:, region_idx, :]
                x1_theta_i = self.transform_theta(x1_theta_i, region_idx)
                x1_trans_i = self.stn(x, x1_theta_i)
                x1_trans_i = F.upsample(x1_trans_i, (24, 28), mode='bilinear', align_corners=True)
                x1_local_i = self.local_conv1(x1_trans_i)
                x1_local_list.append(x1_local_i)

        # ============== Block 2 ==============
        # Block 2
        # global branch
        x2 = self.inception2(x1_out)
        x2_attn, x2_theta = self.ha2(x2)
        x2_out = x2 * x2_attn
        # local branch
        if self.learn_region:
            x2_local_list = []
            for region_idx in range(4):
                x2_theta_i = x2_theta[:, region_idx, :]
                x2_theta_i = self.transform_theta(x2_theta_i, region_idx)
                x2_trans_i = self.stn(x1_out, x2_theta_i)
                x2_trans_i = F.upsample(x2_trans_i, (12, 14), mode='bilinear', align_corners=True)
                x2_local_i = x2_trans_i + x1_local_list[region_idx]
                x2_local_i = self.local_conv2(x2_local_i)
                x2_local_list.append(x2_local_i)

        # ============== Block 3 ==============
        # Block 3
        # global branch
        x3 = self.inception3(x2_out)
        x3_attn, x3_theta = self.ha3(x3)
        x3_out = x3 * x3_attn
        # local branch
        if self.learn_region:
            x3_local_list = []
            for region_idx in range(4):
                x3_theta_i = x3_theta[:, region_idx, :]
                x3_theta_i = self.transform_theta(x3_theta_i, region_idx)
                x3_trans_i = self.stn(x2_out, x3_theta_i)
                x3_trans_i = F.upsample(x3_trans_i, (6, 7), mode='bilinear', align_corners=True)
                x3_local_i = x3_trans_i + x2_local_list[region_idx]
                x3_local_i = self.local_conv3(x3_local_i)
                x3_local_list.append(x3_local_i)

        # ============== Feature generation ==============
        # global branch
        x_global = F.avg_pool2d(x3_out, x3_out.size()[2:]).view(x3_out.size(0), x3_out.size(1))
        x_global = self.fc_global(x_global)
        # local branch
        if self.learn_region:
            x_local_list = []
            for region_idx in range(4):
                x_local_i = x3_local_list[region_idx]
                x_local_i = F.avg_pool2d(x_local_i, x_local_i.size()[2:]).view(x_local_i.size(0), -1)
                x_local_list.append(x_local_i)
            # x_local = torch.cat(x_local_list, 1)
            # x_local = self.fc_local(x_local)
            x_local0=self.fc_local0(x_local_list[0])
            x_local1 = self.fc_local1(x_local_list[1])
            x_local2 = self.fc_local2(x_local_list[2])
            x_local3 = self.fc_local3(x_local_list[3])

        # if not self.training:
        #     # l2 normalization before concatenation
        #     if self.learn_region:
        #         x_global = x_global / x_global.norm(p=2, dim=1, keepdim=True)
        #         x_local = x_local / x_local.norm(p=2, dim=1, keepdim=True)
        #         return torch.cat([x_global, x_local], 1)
        #     else:
        #         return x_global

        prelogits_global = self.classifier_global(x_global)
        if self.learn_region:
            prelogits_local0 = self.classifier_local0(x_local0)
            prelogits_local1 = self.classifier_local1(x_local1)
            prelogits_local2 = self.classifier_local2(x_local2)
            prelogits_local3 = self.classifier_local3(x_local3)

        if self.loss == 'softmax':
            if self.learn_region:
                return (prelogits_global, prelogits_local)
            else:
                return prelogits_global

        elif self.loss == 'triplet':
            if self.learn_region:
                return prelogits_global, x_global,prelogits_local0,prelogits_local1,prelogits_local2,prelogits_local3,x_local0,x_local1,x_local2,x_local3


            else:
                return prelogits_global, x_global

        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))







class MGN(nn.Module):
    def __init__(self):
        super(MGN, self).__init__()

        feats = 256
        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))

        self.reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(self.reduction)

        self.fc_id_2048_0 = nn.Linear(feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)
        ##############################
        self.res_part2=SEResNetBottleNeck(2048,512,groups=1,reduction=16)
        self.part_maxpool=nn.AdaptiveMaxPool2d(1,1)
        self.batch_crop=BatchDrop(height_ratio,width_ratio)
        self.reductionBFE= nn.Sequential(
            nn.Linear(2048,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.reductionBFE.apply(weights_init_kaiming)
        self.softmaxBFE= nn.Linear(256,num_classes)
        self.softmaxBFE.apply(weights_init_kaiming)

        ##############################
        self.haa=HACNN()

        from functools import partial
        import pickle
        pickle.load =partial(pickle.load,encoding='latin1')
        pickle.Unpickler=partial(pickle.Unpickler,encoding='latin1')
        myhaamodel_dict=torch.load('hacnn_market_xent.pth.tar',map_location=lambda storage,loc:storage,pickle_module=pickle)
        model_dict=self.haa.state_dict()
        myhaamodel_dict={k:v for k,v in myhaamodel_dict['state_dict'].items() if 'fc_local' not in k and 'fc_global' not in k and 'classifier_global' not in k and 'classifier_local' not in k}
        model_dict.update(myhaamodel_dict)
        self.haa.load_state_dict(model_dict)




    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x2=F.upsample(x,(160,64),mode='bilinear',align_corners=True)

        prelogits_global, x_global, prelogits_local0, prelogits_local1, prelogits_local2, prelogits_local3, x_local0, x_local1, x_local2, x_local3=self.haa(x2)

        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction(z2_p3).squeeze(dim=3).squeeze(dim=2)

        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        ##########################################
        xbfe=self.res_part2(p4)
        xbfe=self.batch_crop(xbfe)
        triplet_feature=self.part_maxpool(xbfe).squeeze(dim=3).squeeze(dim=2)
        feature=self.reductionBFE(triplet_feature)
        softmax_feature=self.softmaxBFE(feature)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3