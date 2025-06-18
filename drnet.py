import math
import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
import copy
import torch.nn.functional as F


BatchNorm = nn.modules.batchnorm.BatchNorm2d

# __all__ = ['DRN', 'drn26', 'drn42', 'drn58']

webroot = 'https://tigress-web.princeton.edu/~fy/drn/models/'

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'drn-c-26': webroot + 'drn_c_26-ddedf421.pth',
    'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth',
    'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth',
    'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth',
    'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth',
    'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
    'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
}

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
'''conv3x3 函数定义了一个 3x3 的卷积层，其输入参数包括 in_planes（输入平面数），out_planes（输出平面数），stride（步长）、padding（填充）和 dilation（扩张率）。'''
class BasicBlock(nn.Module):
    expansion = 1
    '''BasicBlock 和 Bottleneck 都是继承自 nn.Module 类的子类，并且都实现了 forward 方法。这两个块都包含了若干个卷积层、归一化层和激活函数，用于从输入数据中提取特征。
其中，BasicBlock 中包含了两个 3x3 的卷积层，而 Bottleneck 中包含了三个卷积层（分别是 1x1、3x3 和 1x1）。
这两个块都包含了一个 downsample 参数，用于下采样，以及一个 residual 参数，表示是否需要使用残差连接。如果 residual 为 True，则在块的输出中添加输入数据的残差连接。'''
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual
        if self.downsample is not None:
            pass
        else :
            self.adapterconv = nn.Sequential(
            #卷积层一，做通道数/2，特征图shape/2
            nn.Conv2d(inplanes, int(inplanes/2), kernel_size=(3,3),stride=1,padding=1), 
            nn.ReLU(),
            #卷积层二，特征图shape/2
            nn.Conv2d(int(inplanes/2), inplanes,  kernel_size=(3,3),stride=1,padding=1), )
            # nn.Conv2d(inplanes,int(inplanes/2),kernel_size=(3,3),stride=1,padding=2),
            # nn.MaxPool2d(2,2),
            # nn.BatchNorm2d(int(inplanes/2)),
            # nn.ReLU(),
            # nn.Conv2d(int(inplanes/2),int(inplanes/2),kernel_size=(3,3),stride=1,padding=2),
            # nn.MaxPool2d(2,2),
            # nn.BatchNorm2d(int(inplanes/2)),
            # nn.ReLU(),
            # nn.ConvTranspose2d(int(inplanes/2),int(inplanes/2),(1,1),stride=2),
            # nn.BatchNorm2d(int(inplanes/2)),
            # nn.ReLU(),
            # nn.ConvTranspose2d(int(inplanes/2),inplanes,(1,1),stride=2)
            
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        if self.downsample is not None:
            
            residual = self.downsample(x)
        else :
            adapterconv = self.adapterconv(x)
            if adapterconv.shape[2]%2 != 0:
                adapterconv = adapterconv[:,:,:-1,:] #对第三维度输出非偶数的维度不匹配现象，舍去adconv第三维度最后一行向量
            out += adapterconv
        if self.residual: ## Noresidual connection in degridding networks
            out += residual
       
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True,layernum = -1, useadapter = True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.useadapter = useadapter
        self.layernum = layernum
        # print(self.layernum)
        # print(self.downsample is not None)
        if self.downsample is not None:
            pass
        else :
            if useadapter :
                if layernum >=5:
                    self.adapterconv = nn.Sequential(
                    #卷积层一，做通道数/2，特征图shape/2
                    nn.Conv2d(inplanes, int(inplanes/4), kernel_size=(3,3),stride=1,padding=1), 
                    nn.ReLU(),
                    #卷积层二，特征图shape/2
                    nn.Conv2d(int(inplanes/4), inplanes,  kernel_size=(3,3),stride=1,padding=1), )
            else :
                pass
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        #out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        else :
            if self.useadapter :
                if self.layernum  >=5 :
                    adapterconv = self.adapterconv(x)
                    if adapterconv.shape[2]%2 != 0:
                        adapterconv = adapterconv[:,:,:-1,:] #对第三维度输出非偶数的维度不匹配现象，舍去adconv第三维度最后一行向量
                    out += adapterconv
                    #print(f"Adapter in layer{self.layernum}")
        out += residual ## Always there is a residual connection
        out = self.relu(out)

        return out

class DRN(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 channels=(16, 32, 64, 64, 64, 64, 64, 64),
                 out_map=False, out_middle=False, pool_size=28, arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False)
            self.bn1 = BatchNorm(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                BatchNorm(channels[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_conv_layers(channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(channels[1], layers[1], stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2,numlayer = 3)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2,numlayer = 4)
        self.layer5 = self._make_layer(block, channels[4], layers[4],dilation=2, new_level=False,numlayer = 5)
        self.layer6 = None if layers[5] == 0 else \
            self._make_layer(block, channels[5], layers[5], dilation=4, new_level=False,numlayer = 6)

        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else \
                self._make_layer(BasicBlock, channels[6], layers[6], dilation=2, new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else \
                self._make_layer(BasicBlock, channels[7], layers[7], dilation=1, new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else \
                self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else \
                self._make_conv_layers(channels[7], layers[7], dilation=1)

        if num_classes > 0:
            self.avgpool = nn.AvgPool2d(1)
            self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, new_level=True, residual=True,numlayer = -1):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(
              block(self.inplanes, planes, stride, downsample,
                dilation=(1, 1) if dilation == 1 else (dilation // 2 if new_level else dilation, dilation),
                residual=residual,layernum = numlayer
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                   block(self.inplanes, planes, residual=residual, dilation=(dilation, dilation))
              )

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()

        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)

        x = self.layer3(x)
        y.append(x)

        x = self.layer4(x)
        y.append(x)

        x = self.layer5(x)
        y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)

        if self.out_map:
            x = self.fc(x)
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

        if self.out_middle:
            return x, y
        else:
            return x

class DRN_A(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(DRN_A, self).__init__()
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.avgpool = nn.AvgPool2d(28, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                   block(self.inplanes, planes, dilation=(dilation, dilation))
              )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def drn_a_50(pretrained=False, **kwargs):
    model = DRN_A(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def drn_c_26(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-26']))
    return model

def drn_c_42(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-42']))
    return model

def drn_c_58(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-c-58']))
    return model

def drn_d_22(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-22']))
    return model
def drn_d_14(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 1, 1, 1, 1, 1, 1], arch='D', **kwargs)
    if pretrained:
        print("无可下载的预训练权重")
        #model.load_state_dict(model_zoo.load_url(model_urls['drn-d-22']))
    return model

def drn_d_24(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-24']))
    return model

def drn_d_38(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-38']))
        print("Loading pretrained model on ImageNet")
    return model

def drn_d_40(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-40']))
    return model

def drn_d_54(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-54']))
    return model

def drn_d_56(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-56']))
    return model

def drn_d_105(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-105']))
    return model

def drn_d_107(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 2, 2], arch='D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-107']))
    return model

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def conv_init(m):
	n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
	m.weight.data.normal_(0, math.sqrt(2. / n))		
	m.bias.data.zero_()

class Net(nn.Module):
    def __init__(self, classes, embed_dim, resnet, pretrained_model=None, pretrained=None, use_torch_up=False):
        super().__init__()
        assert(isinstance(classes , dict)), f"num_labels should be dict, got {type(classes)}"
      

        self.datasets = list(classes.keys())
        self.embed_dim = embed_dim

        resnet_archs = {'resnet_18':drn_d_22 , 'resnet_34':drn_d_38, 'resnet_50':drn_d_54 , 'resnet_101':drn_d_105}
        arch = resnet_archs[resnet]
        
        model = arch(pretrained=pretrained, num_classes=1000)
        #pmodel = nn.DataParallel(model)
        #if pretrained_model is not None:
        #	pmodel.load_state_dict(pretrained_model)

        self.base = nn.Sequential(*list(model.children())[:-2]) ## Encoder. 
        self.seg = nn.ModuleList() ## Decoder 1d conv
        self.up = nn.ModuleList() ## Decoder upsample (non-trainable)

        self.net1 = nn.ConvTranspose2d(64,64, (16,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
        fill_up_weights(self.net1)
        self.net2 = nn.ConvTranspose2d(64,64, (9,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
        fill_up_weights(self.net2)
        self.net3 = nn.Conv2d(64,16, kernel_size=1, bias=True)
        conv_init(self.net3)
        self.net4 = nn.Conv2d(64,16, kernel_size=1, bias=True)
        conv_init(self.net4)
        self.net5 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net5)
        self.net6 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net6)
        self.net7 = nn.Sequential(
				nn.Linear(832,64),#832,64
				#nn.Dropout(0.2),
				nn.ReLU(),
				nn.Linear(64,8))#8
        self.net8 = nn.Sequential(
				nn.Linear(417,64),#417,64
				#nn.Dropout(0.1),
				nn.ReLU(),
				nn.Linear(64,8))
        '''self.net7 = nn.ModuleDict([[ d , nn.Sequential(
				nn.Linear(200,64),
				#nn.Dropout(0.2),
				nn.ReLU(),
				nn.Linear(64,4))] for d in self.datasets])
		self.net8 = nn.ModuleDict([[ d , nn.Sequential(
				nn.Linear(417,64),
				#nn.Dropout(0.1),
				nn.ReLU(),
				nn.Linear(64,4))] for d in self.datasets])'''
        '''self.net7 = nn.Linear(830,64)
		self.net8 = nn.Linear(417,64)
		self.net9 = nn.Linear(64,4)
		self.net10 = nn.Linear(64,4)'''
        #self.x_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(25,53), bias=True))#104,1
        self.x_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(104,1), bias=True))#104,1
        self.y_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(1,53), bias=True))

    def forward(self, x):
        encoder = self.base(x) #resnet-18

        axis_y = self.y_pre(encoder) #Size compression
        axis_y = self.net1(axis_y)    #transpose convolution
        axis_y = self.net3(axis_y)   #1x1 convolution
        axis_y = nn.ReLU()(axis_y)
        axis_y = self.net5(axis_y)
        axis_y = nn.ReLU()(axis_y)
        '''axis_y = self.net7(axis_y.view(axis_y.shape[0],-1)) 
        axis_y = nn.ReLU()(axis_y)
        coord_y = self.net9(axis_y)'''
        #coord_y = { d : self.net7[d](axis_y.view(axis_y.shape[0],-1)) for d in self.datasets}
        x=axis_y.view(axis_y.shape[0],-1)
        #coord_y = self.net7(axis_y.view(axis_y.shape[0],-1))
        coord_y = self.net7(x)

        axis_x = self.x_pre(encoder)
        axis_x = self.net2(axis_x)
        axis_x = self.net4(axis_x)
        axis_x = nn.ReLU()(axis_x)
        axis_x = self.net6(axis_x)
        axis_x = nn.ReLU()(axis_x)
        '''axis_x = self.net8(axis_x.view(axis_x.shape[0],-1)) 
        axis_x = nn.ReLU()(axis_x)
        coord_x = self.net10(axis_x)'''
        #coord_x = { d : self.net8[d](axis_x.view(axis_x.shape[0],-1)) for d in self.datasets}
        z = axis_x.view(axis_x.shape[0],-1)
        coord_x = self.net8(z)

        #output = {d : torch.cat((coord_x[d].view(-1,4,1),coord_y[d].view(-1,4,1)), 2) for d in self.datasets}
        output = torch.cat((coord_x.view(-1,8,1),coord_y.view(-1,8,1)), 2)

        return output
    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

class Net_withAE(nn.Module):

    def __init__(self, classes, embed_dim, resnet, pretrained_model=None, pretrained=None, use_torch_up=False, use_convAE = True, ):
        super().__init__()
        assert(isinstance(classes , dict)), f"num_labels should be dict, got {type(classes)}"
      
        self.use_convAE = use_convAE
        self.datasets = list(classes.keys())
        self.embed_dim = embed_dim

        resnet_archs = {'resnet_18':drn_d_22 , 'resnet_34':drn_d_38, 'resnet_50':drn_d_54 , 'resnet_101':drn_d_105}
        arch = resnet_archs[resnet]
        
        model = arch(pretrained=pretrained, num_classes=1000)

        if use_convAE:    
            self.AE_X = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,3), stride=1),  # 训练好的卷积encoder提取特征
            nn.ReLU(),
            nn.MaxPool2d(1,2),  # 最大池化后shape/2
            )

            self.AE_Y = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,1), stride=1,padding=(1,0)),  # 输入通道数为1，输出通道数为16
            nn.ReLU(),
            nn.MaxPool2d(1,2),  # 输入通道数为16，输出通道数为8
        )
            # weights_X = torch.load('AE_save/featureX_dict_beforeconv_weights.pth', map_location=torch.device('cpu'))
            # weights_Y = torch.load('AE_save/featureY_dict_beforeconv_weights.pth', map_location=torch.device('cpu'))
            # #selected_state_dict = {k: v for k, v in weights_X.items() if k in model_dict}
            # encoder_weights = {}
            # for k,v in weights_X.items():
            #     if "encoder" in k:
            #         encoder_weights[k] = v
            # self.encoder.load_state_dict(encoder_weights)
                    


        else:

            embid_dim = 150
            self.AE_X = nn.Sequential(
            nn.Linear(417, embid_dim),
            #nn.ReLU(),
            #nn.Dropout(0.2)  # 添加 Dropout 层，丢弃比例为 0.2
            )

            self.AE_Y = nn.Sequential(
            nn.Linear(832, embid_dim),
            #nn.ReLU(),
            #nn.Dropout(0.2)  # 添加 Dropout 层，丢弃比例为 0.2
            )


        
        #pmodel = nn.DataParallel(model)
        #if pretrained_model is not None:
        #	pmodel.load_state_dict(pretrained_model)

        self.base = nn.Sequential(*list(model.children())[:-2]) ## Encoder. 
        self.seg = nn.ModuleList() ## Decoder 1d conv
        self.up = nn.ModuleList() ## Decoder upsample (non-trainable)

        self.net1 = nn.ConvTranspose2d(64,64, (16,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
        fill_up_weights(self.net1)
        self.net2 = nn.ConvTranspose2d(64,64, (9,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
        fill_up_weights(self.net2)
        self.net3 = nn.Conv2d(128 if use_convAE else 64,16, kernel_size=1, bias=True)
        conv_init(self.net3)
        self.net4 = nn.Conv2d(128 if use_convAE else 64,16, kernel_size=1, bias=True)
        conv_init(self.net4)
        self.net5 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net5)
        self.net6 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net6)
        self.net7 = nn.Sequential(
				nn.Linear(int(832/2) if use_convAE else embed_dim,64),#832,64
				#nn.Dropout(0.2),
				nn.ReLU(),
				nn.Linear(64,8))#8
        self.net8 = nn.Sequential(
				nn.Linear(208 if use_convAE else embid_dim,64),#417,64
				#nn.Dropout(0.1),
				nn.ReLU(),


				nn.Linear(64,8))
        '''self.net7 = nn.ModuleDict([[ d , nn.Sequential(
				nn.Linear(200,64),
				#nn.Dropout(0.2),
				nn.ReLU(),
				nn.Linear(64,4))] for d in self.datasets])
		self.net8 = nn.ModuleDict([[ d , nn.Sequential(
				nn.Linear(417,64),
				#nn.Dropout(0.1),
				nn.ReLU(),
				nn.Linear(64,4))] for d in self.datasets])'''
        '''self.net7 = nn.Linear(830,64)
		self.net8 = nn.Linear(417,64)
		self.net9 = nn.Linear(64,4)
		self.net10 = nn.Linear(64,4)'''
        #self.x_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(25,53), bias=True))#104,1
        self.x_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(104,1), bias=True))#104,1
        self.y_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(1,53), bias=True))

    def forward(self, x):
        encoder = self.base(x) #resnet-18

        axis_y = self.y_pre(encoder) #Size compression
        axis_y = self.net1(axis_y)    #transpose convolution
        if self.use_convAE:
            axis_y = self.AE_Y(axis_y)
        axis_y = self.net3(axis_y)   #1x1 convolution
        axis_y = nn.ReLU()(axis_y)
        axis_y = self.net5(axis_y)
        axis_y = nn.ReLU()(axis_y)
        '''axis_y = self.net7(axis_y.view(axis_y.shape[0],-1)) 
        axis_y = nn.ReLU()(axis_y)
        coord_y = self.net9(axis_y)'''
        #coord_y = { d : self.net7[d](axis_y.view(axis_y.shape[0],-1)) for d in self.datasets}
        x=axis_y.view(axis_y.shape[0],-1)
        if not self.use_convAE:
            x = self.AE_Y(x)
        #coord_y = self.net7(axis_y.view(axis_y.shape[0],-1))
        coord_y = self.net7(x)

        axis_x = self.x_pre(encoder)
        axis_x = self.net2(axis_x)
        if self.use_convAE:
            axis_x = self.AE_X(axis_x)
        axis_x = self.net4(axis_x)
        axis_x = nn.ReLU()(axis_x)
        axis_x = self.net6(axis_x)
        axis_x = nn.ReLU()(axis_x)
        '''axis_x = self.net8(axis_x.view(axis_x.shape[0],-1)) 
        axis_x = nn.ReLU()(axis_x)
        coord_x = self.net10(axis_x)'''
        #coord_x = { d : self.net8[d](axis_x.view(axis_x.shape[0],-1)) for d in self.datasets}
        z = axis_x.view(axis_x.shape[0],-1)
        if not self.use_convAE:
            z = self.AE_X(z)
        coord_x = self.net8(z)

        #output = {d : torch.cat((coord_x[d].view(-1,4,1),coord_y[d].view(-1,4,1)), 2) for d in self.datasets}
        output = torch.cat((coord_x.view(-1,8,1),coord_y.view(-1,8,1)), 2)

        return output
    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param



class Net_FPN(nn.Module):
    def __init__(self, classes, embed_dim, resnet, pretrained_model=None, pretrained=None, use_torch_up=False):
        super().__init__()
        assert(isinstance(classes , dict)), f"num_labels should be dict, got {type(classes)}"
        self.datasets = list(classes.keys())
        self.embed_dim = embed_dim

        resnet_archs = {'resnet_18':drn_d_22 , 'resnet_34':drn_d_38, 'resnet_50':drn_d_54 , 'resnet_101':drn_d_105}
        arch = resnet_archs[resnet]
        
        model = arch(pretrained=pretrained, num_classes=1000)
        #pmodel = nn.DataParallel(model)
        #if pretrained_model is not None:
        #	pmodel.load_state_dict(pretrained_model)

        #self.base = nn.Sequential(*list(model.children())[:-2]) ## Encoder. 
        self.layer1 = nn.Sequential(*list(model.children())[0])
        self.myx0 = nn.Conv2d(16, 64, kernel_size=1)

        self.layer2 = nn.Sequential(*list(model.children())[1])
        self.myx1 = nn.Conv2d(16, 64, kernel_size=1)

        self.layer3 = nn.Sequential(*list(model.children())[2])
        self.myx2 = nn.Conv2d(32, 64, kernel_size=1)

        self.layer4 = nn.Sequential(*list(model.children())[3])
        self.myx3 = nn.Conv2d(64, 64, kernel_size=1)


        self.layer5 = nn.Sequential(*list(model.children())[4])
        self.myx4 = nn.Conv2d(64, 64, kernel_size=1)

        self.layer6 = nn.Sequential(*list(model.children())[5])
        self.myx5 = nn.Conv2d(64, 64, kernel_size=1)

        self.layer7 = nn.Sequential(*list(model.children())[6])
        self.myx6 = nn.Conv2d(64, 64, kernel_size=1)

        self.dropout = nn.Dropout(p=0.2)  # dropout训练

        self.layer8 = nn.Sequential(*list(model.children())[7])
        self.channel_adjustment = nn.Conv2d(384, 64, kernel_size=1)
        self.seg = nn.ModuleList() ## Decoder 1d conv
        self.up = nn.ModuleList() ## Decoder upsample (non-trainable)

        self.net1 = nn.ConvTranspose2d(64,64, (16,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
 

        fill_up_weights(self.net1)
        self.net2 = nn.ConvTranspose2d(64,64, (9,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
        fill_up_weights(self.net2)
        self.net3 = nn.Conv2d(64,16, kernel_size=1, bias=True)
        conv_init(self.net3)
        self.net4 = nn.Conv2d(64,16, kernel_size=1, bias=True)
        conv_init(self.net4)
        self.net5 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net5)
        self.net6 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net6)
        self.net7 = nn.Sequential(
				nn.Linear(832,64),#200,64
				#nn.Dropout(0.2),
				nn.ReLU(),
				nn.Linear(64,8))#8
        self.net8 = nn.Sequential(
				nn.Linear(417,64),#1,64
				#nn.Dropout(0.1),
				nn.ReLU(),
				nn.Linear(64,8))

        self.x_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(104,1), bias=True))#25,53
        self.y_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(1,53), bias=True))

    def forward(self, x):
        #encoder = self.base(x) #resnet-18,即featuremap
        y = list()
                
        x = self.layer1(x)
        X0 = self.myx0(x)
        y.append(X0)
        x = self.layer2(x)
        X1 = self.myx1(x)

        y.append(X1)

        x = self.layer3(x)
        X2 = self.myx2(x)

        y.append(X2)

        x = self.layer4(x)
        X3 = self.myx3(x)

        y.append(X3)

        x = self.layer5(x)
        X4 = self.myx4(x)

        y.append(X4)

        x = self.layer6(x)
        X5 = self.myx5(x)

        y.append(X5)

        x = self.layer7(x)
        X6 = self.myx6(x)

        y.append(X6)
        # for num, i in enumerate(y) :
        #     print(f"第{num}个元素的shape为{i.shape}")
        resized_layer1 = F.interpolate(y[0], size=(104,53), mode='bilinear', align_corners=False)
        resized_layer2 = F.interpolate(y[1], size=(104,53), mode='bilinear', align_corners=False)
        resized_layer3 = F.interpolate(y[2], size=(104,53), mode='bilinear', align_corners=False)
        resized_layer4 = F.interpolate(y[3], size=(104,53), mode='bilinear', align_corners=False)
        resized_layer5 = F.interpolate(y[4], size=(104,53), mode='bilinear', align_corners=False)
        resized_layer6 = F.interpolate(y[5], size=(104,53), mode='bilinear', align_corners=False)
        resized_layer7 = F.interpolate(y[6], size=(104,53), mode='bilinear', align_corners=False)
        x = (x + resized_layer1 +resized_layer2 + resized_layer3 + resized_layer4 + resized_layer5 +resized_layer6 + resized_layer7)/8
        x = self.dropout(x)
        #x = (sum).mean(dim = (-1, -2))

        #x = torch.cat([x, resized_layer1, resized_layer2, resized_layer3, resized_layer4, resized_layer5, resized_layer6, resized_layer7], dim=1)
        

        encoder = self.layer8(x)
        y.append(x)


        
        axis_y = self.y_pre(encoder) #Size compression
        # axis_y = self.chuanliany1(axis_y)
        # axis_y = self.chuanliany2(axis_y)
        axis_y = self.net1(axis_y)    #transpose convolution




        # axis_y_branch = axis_y.view(axis_y.shape[0], axis_y.shape[1], -1)  # 将输入展平为 (batch_size, channels, length * width)

        # axis_y_branch = self.fc_ydownencoder(axis_y_branch)  # 下采样
        # axis_y_branch = self.bn_ydownencoder(axis_y_branch)
        # axis_y_branch = torch.relu(axis_y_branch)
        # axis_y_branch = self.fc_yupdecoder(axis_y_branch)  # 上采样
        # axis_y_branch = self.bn_yupencoder(axis_y_branch)
        # axis_y_branch = axis_y_branch.unsqueeze(-1) # 将输出重新调整为原始形状 (batch_size, channels, length, width)
            
        # axis_y = axis_y + 0.8*axis_y_branch
        # axis_y = self.yencoder1(axis_y)
        # #print(9999)
        # axis_y = self.yencoder2(axis_y)
        # axis_y = self.yencoder3(axis_y)
        # axis_y = self.ydecoder1(axis_y)
        # axis_y = self.ydecoder2(axis_y)
        # axis_y = self.ydecoder3(axis_y)




        axis_y = self.net3(axis_y)   #1x1 convolution
        axis_y = nn.ReLU()(axis_y)
        axis_y = self.net5(axis_y)#ytezheng 
        axis_y = nn.ReLU()(axis_y)
        
        '''axis_y = self.net7(axis_y.view(axis_y.shape[0],-1)) 
        axis_y = nn.ReLU()(axis_y)
        coord_y = self.net9(axis_y)'''
        #coord_y = { d : self.net7[d](axis_y.view(axis_y.shape[0],-1)) for d in self.datasets}
        x=axis_y.view(axis_y.shape[0],-1)
        #coord_y = self.net7(axis_y.view(axis_y.shape[0],-1))
        coord_y = self.net7(x)

        axis_x = self.x_pre(encoder)

        # axis_x = self.chuanlianx1(axis_x)
        # axis_x = self.chuanlianx2(axis_x)




        axis_x = self.net2(axis_x)

        # axis_x_branch = axis_x.view(axis_x.shape[0], axis_x.shape[1], -1)  # 将输入展平为 (batch_size, channels, length * width)
        # axis_x_branch = self.fc_xdownencoder(axis_x_branch)
        # axis_x_branch = self.bn_xdownencoder(axis_x_branch)
        # axis_x_branch = torch.relu(axis_x_branch)
        # axis_x_branch = self.fc_xupdecoder(axis_x_branch)
        # axis_x_branch = self.bn_xupencoder(axis_x_branch)
        # axis_x_branch = axis_x_branch.unsqueeze(-1)
        # axis_x_branch = axis_x_branch.permute(0, 1, 3, 2)
        # axis_x = axis_x + 0.8*axis_x_branch
        # axis_x = self.xencoder1(axis_x)
        # axis_x = self.xencoder2(axis_x)
        # axis_x = self.xencoder3(axis_x)
        # axis_x = self.xdecoder1(axis_x)
        # axis_x = self.xdecoder2(axis_x)
        # axis_x = self.xdecoder3(axis_x)
       

        # axis_x = self.xdownsample(axis_x)  # 下采样
        # axis_x = self.xnonlinear(axis_x)  # 非线性变换
        # axis_x = self.xupsample(axis_x)  # 上采样
        axis_x = self.net4(axis_x)
        axis_x = nn.ReLU()(axis_x)
        axis_x = self.net6(axis_x)#xtezheng canshujicheng
        axis_x = nn.ReLU()(axis_x)
        
        '''axis_x = self.net8(axis_x.view(axis_x.shape[0],-1)) 
        axis_x = nn.ReLU()(axis_x)
        coord_x = self.net10(axis_x)'''
        #coord_x = { d : self.net8[d](axis_x.view(axis_x.shape[0],-1)) for d in self.datasets}
        coord_x = self.net8(axis_x.view(axis_x.shape[0],-1))

        #output = {d : torch.cat((coord_x[d].view(-1,4,1),coord_y[d].view(-1,4,1)), 2) for d in self.datasets}
        output = torch.cat((coord_x.view(-1,8,1),coord_y.view(-1,8,1)), 2)

        return output

    def optim_parameters(self, memo=None):
        # for param in self.base.parameters():
        #     yield param
        for param in self.layer1.parameters():
            yield param
        for param in self.layer2.parameters():
            yield param
        for param in self.layer3.parameters():
            yield param
        for param in self.layer4.parameters():
            yield param
        for param in self.layer5.parameters():
            yield param
        for param in self.layer6.parameters():
            yield param
        for param in self.layer7.parameters():
            yield param
        for param in self.layer8.parameters():
            yield param
        
        for param in self.seg.parameters():
            yield param

class Net_transfer(nn.Module):
    def __init__(self, classes, embed_dim, resnet, pretrained_model=None, pretrained=None, use_torch_up=False):
        super().__init__()
        assert(isinstance(classes , dict)), f"num_labels should be dict, got {type(classes)}"
        self.datasets = list(classes.keys())
        self.embed_dim = embed_dim

        resnet_archs = {'resnet_18':drn_d_22 , 'resnet_34':drn_d_38, 'resnet_50':drn_d_54 , 'resnet_101':drn_d_105}
        arch = resnet_archs[resnet]
        
        model = arch(pretrained=pretrained, num_classes=1000)
        #pmodel = nn.DataParallel(model)
        #if pretrained_model is not None:
        #	pmodel.load_state_dict(pretrained_model)

        self.base = nn.Sequential(*list(model.children())[:-2]) ## Encoder. 
        self.seg = nn.ModuleList() ## Decoder 1d conv
        self.up = nn.ModuleList() ## Decoder upsample (non-trainable)

        self.net1 = nn.ConvTranspose2d(64,64, (16,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
 
        # self.ydownsample = nn.Conv2d(64, 64, kernel_size=1, stride=2, padding=1, bias=False)  # 下采样卷积操作
        # self.ynonlinear = nn.ReLU()  # 非线性激活函数
        # self.yupsample = nn.ConvTranspose2d(64, 64, kernel_size=1, stride=2, padding=1, output_padding=1, bias=False)

        

        fill_up_weights(self.net1)
        self.net2 = nn.ConvTranspose2d(64,64, (9,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
        fill_up_weights(self.net2)
        self.net3 = nn.Conv2d(64,16, kernel_size=1, bias=True)
        conv_init(self.net3)
        self.net4 = nn.Conv2d(64,16, kernel_size=1, bias=True)
        conv_init(self.net4)
        self.net5 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net5)
        self.net6 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net6)
        self.net7 = nn.Sequential(
				nn.Linear(832,64),#200,64
				#nn.Dropout(0.2),
				nn.ReLU(),
				nn.Linear(64,8))#8
        self.net8 = nn.Sequential(
				nn.Linear(417,64),#1,64
				#nn.Dropout(0.1),
				nn.ReLU(),
				nn.Linear(64,8))
        '''self.net7 = nn.ModuleDict([[ d , nn.Sequential(
				nn.Linear(200,64),
				#nn.Dropout(0.2),
				nn.ReLU(),
				nn.Linear(64,4))] for d in self.datasets])
		self.net8 = nn.ModuleDict([[ d , nn.Sequential(
				nn.Linear(417,64),
				#nn.Dropout(0.1),
				nn.ReLU(),
				nn.Linear(64,4))] for d in self.datasets])'''
        '''self.net7 = nn.Linear(830,64)
		self.net8 = nn.Linear(417,64)
		self.net9 = nn.Linear(64,4)
		self.net10 = nn.Linear(64,4)'''
        self.x_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(104,1), bias=True))#25,53
        self.y_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(1,53), bias=True))

    def forward(self, x):
        encoder = self.base(x) #resnet-18,即featuremap

        axis_y = self.y_pre(encoder) #Size compression
        axis_y = self.net1(axis_y)    #transpose convolution


        axis_y = self.net3(axis_y)   #1x1 convolution
        axis_y = nn.ReLU()(axis_y)
        axis_y = self.net5(axis_y)#ytezheng 
        axis_y = nn.ReLU()(axis_y)
        '''axis_y = self.net7(axis_y.view(axis_y.shape[0],-1)) 
        axis_y = nn.ReLU()(axis_y)
        coord_y = self.net9(axis_y)'''
        #coord_y = { d : self.net7[d](axis_y.view(axis_y.shape[0],-1)) for d in self.datasets}
        x=axis_y.view(axis_y.shape[0],-1)
        #coord_y = self.net7(axis_y.view(axis_y.shape[0],-1))
        coord_y = self.net7(x)

        axis_x = self.x_pre(encoder)
        axis_x = self.net2(axis_x)
        # axis_x = self.xdownsample(axis_x)  # 下采样
        # axis_x = self.xnonlinear(axis_x)  # 非线性变换
        # axis_x = self.xupsample(axis_x)  # 上采样
        axis_x = self.net4(axis_x)
        axis_x = nn.ReLU()(axis_x)
        axis_x = self.net6(axis_x)#xtezheng canshujicheng
        axis_x = nn.ReLU()(axis_x)
        '''axis_x = self.net8(axis_x.view(axis_x.shape[0],-1)) 
        axis_x = nn.ReLU()(axis_x)
        coord_x = self.net10(axis_x)'''
        #coord_x = { d : self.net8[d](axis_x.view(axis_x.shape[0],-1)) for d in self.datasets}
        coord_x = self.net8(axis_x.view(axis_x.shape[0],-1))

        #output = {d : torch.cat((coord_x[d].view(-1,4,1),coord_y[d].view(-1,4,1)), 2) for d in self.datasets}
        output = torch.cat((coord_x.view(-1,8,1),coord_y.view(-1,8,1)), 2)

        return output

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param



    def __init__(self, classes, embed_dim, resnet, pretrained_model=None, pretrained=None, use_torch_up=False):
        super().__init__()
        assert(isinstance(classes , dict)), f"num_labels should be dict, got {type(classes)}"
        self.datasets = list(classes.keys())
        self.embed_dim = embed_dim

        resnet_archs = {'resnet_18':drn_d_22 , 'resnet_34':drn_d_38, 'resnet_50':drn_d_54 , 'resnet_101':drn_d_105}
        arch = resnet_archs[resnet]
        
        model = arch(pretrained=pretrained, num_classes=1000)
        #pmodel = nn.DataParallel(model)
        #if pretrained_model is not None:
        #	pmodel.load_state_dict(pretrained_model)

        self.base = nn.Sequential(*list(model.children())[:-2]) ## Encoder. 

  

        self.seg = nn.ModuleList() ## Decoder 1d conv
        self.up = nn.ModuleList() ## Decoder upsample (non-trainable)

        self.net1 = nn.ConvTranspose2d(64,64, (16,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
 
        # self.ydownsample = nn.Conv2d(64, 64, kernel_size=1, stride=2, padding=1, bias=False)  # 下采样卷积操作
        # self.ynonlinear = nn.ReLU()  # 非线性激活函数
        # self.yupsample = nn.ConvTranspose2d(64, 64, kernel_size=1, stride=2, padding=1, output_padding=1, bias=False)

        self.yencoder = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=(3,1), stride=1,padding=(1,0)),  # 输入通道数为1，输出通道数为16
            #nn.MaxPool2d(1,2),  # 输入通道数为16，输出通道数为8
            nn.ReLU(),
            # nn.Conv2d(16, 8, kernel_size=(3,1), stride=1,padding=(1,0)),  # 输入通道数为1，输出通道数为16
            # nn.MaxPool2d(1,2),  # 输入通道数为16，输出通道数为8
            # nn.ReLU(),
            
            
        )
        
        self.ydecoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=(3,1), stride=1,padding=(1,0)),  # 输入通道数为128，输出通道数为64
            #nn.ConvTranspose2d(16, 64, kernel_size=(3,1), stride=2,padding=(1,0),output_padding=(1,0)),  # 输入通道数为128，输出通道数为64

        )
        self.xencoder = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=(1,3), stride=1,padding=(0,1)),  # 输入通道数为1，输出通道数为16
            #nn.MaxPool2d(1,2),  # 输入通道数为16，输出通道数为8
            nn.ReLU(),
            # nn.Conv2d(16, 8, kernel_size=(1,3), stride=1,padding=(0,1)),  # 输入通道数为1，输出通道数为16
            # nn.MaxPool2d(1,2),  # 输入通道数为16，输出通道数为8
            # nn.ReLU(),
        )
        
        self.xdecoder = nn.Sequential(
            #nn.ConvTranspose2d(8, 16, kernel_size=(1,3), stride=1,padding=(0,1)),  # 输入通道数为128，输出通道数为64
            #nn.ConvTranspose2d(16, 64, kernel_size=(1,3), stride=2,padding=(0,1)),  # 输入通道数为128，输出通道数为64
            nn.Conv2d(8, 16, kernel_size=(1,3), stride=1,padding=(0,1)),  # 输入通道数为1，输出通道数为16

        )

        fill_up_weights(self.net1)
        self.net2 = nn.ConvTranspose2d(64,64, (9,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
        fill_up_weights(self.net2)
        self.net3 = nn.Conv2d(64,16, kernel_size=1, bias=True)
        conv_init(self.net3)
        self.net4 = nn.Conv2d(64,16, kernel_size=1, bias=True)
        conv_init(self.net4)
        self.net5 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net5)
        self.net6 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net6)
        self.net7 = nn.Sequential(
				nn.Linear(832,64),#200,64
				#nn.Dropout(0.2),
				nn.ReLU(),
				nn.Linear(64,8))#8
        self.net8 = nn.Sequential(
				nn.Linear(417,64),#1,64
				#nn.Dropout(0.1),
				nn.ReLU(),
				nn.Linear(64,8))

        self.x_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(104,1), bias=True))#25,53
        self.y_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(1,53), bias=True))

    def forward(self, x):
        encoder = self.base(x) #resnet-18,即featuremap

        axis_y = self.y_pre(encoder) #Size compression
        axis_y = self.net1(axis_y)    #transpose convolution

        axis_y_branch = self.yencoder(axis_y)
        axis_y_branch = self.ydecoder(axis_y_branch)
        axis_y = self.net3(axis_y)   #1x1 convolution
        #axis_y = axis_y_branch +axis_y#加分支上的可学习卷积层
        axis_y = axis_y
        axis_y = nn.ReLU()(axis_y) 
        axis_y = self.net5(axis_y)#ytezheng 
        axis_y = nn.ReLU()(axis_y)

        x=axis_y.view(axis_y.shape[0],-1)

        coord_y = self.net7(x)

        axis_x = self.x_pre(encoder)
        axis_x = self.net2(axis_x)
        axis_x_branch= self.xencoder(axis_x)
        axis_x_branch = self.xdecoder(axis_x_branch)

        axis_x = self.net4(axis_x)
        #axis_x = axis_x + axis_x_branch#加分支上的可学习卷积层
        axis_x = nn.ReLU()(axis_x) 
        axis_x = self.net6(axis_x)#xtezheng canshujicheng
        axis_x = nn.ReLU()(axis_x)

        coord_x = self.net8(axis_x.view(axis_x.shape[0],-1))


        output = torch.cat((coord_x.view(-1,8,1),coord_y.view(-1,8,1)), 2)

        return output

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

class Net_transfer_SkipConnection(nn.Module):
    def __init__(self, classes, embed_dim, resnet, pretrained_model=None, pretrained=None, use_torch_up=False):
        super().__init__()
        assert(isinstance(classes , dict)), f"num_labels should be dict, got {type(classes)}"
        self.datasets = list(classes.keys())
        self.embed_dim = embed_dim

        resnet_archs = {'resnet_18':drn_d_22 , 'resnet_34':drn_d_38, 'resnet_50':drn_d_54 , 'resnet_101':drn_d_105}
        arch = resnet_archs[resnet]
        
        model = arch(pretrained=pretrained, num_classes=1000)
        #pmodel = nn.DataParallel(model)
        #if pretrained_model is not None:
        #	pmodel.load_state_dict(pretrained_model)

        self.base = nn.Sequential(*list(model.children())[:-2]) ## Encoder. 

  

        self.seg = nn.ModuleList() ## Decoder 1d conv
        self.up = nn.ModuleList() ## Decoder upsample (non-trainable)

        

        self.net1 = nn.ConvTranspose2d(64,64, (16,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
 
        # self.ydownsample = nn.Conv2d(64, 64, kernel_size=1, stride=2, padding=1, bias=False)  # 下采样卷积操作
        # self.ynonlinear = nn.ReLU()  # 非线性激活函数
        # self.yupsample = nn.ConvTranspose2d(64, 64, kernel_size=1, stride=2, padding=1, output_padding=1, bias=False)

        self.yencoder = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=(3,1), stride=1,padding=(1,0)),  # 输入通道数为1，输出通道数为16
            #nn.MaxPool2d(1,2),  # 输入通道数为16，输出通道数为8
            nn.ReLU(),
            # nn.Conv2d(16, 8, kernel_size=(3,1), stride=1,padding=(1,0)),  # 输入通道数为1，输出通道数为16
            # nn.MaxPool2d(1,2),  # 输入通道数为16，输出通道数为8
            # nn.ReLU(),
            
            
        )
        
        self.ydecoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=(3,1), stride=1,padding=(1,0)),  # 输入通道数为128，输出通道数为64
            #nn.ConvTranspose2d(16, 64, kernel_size=(3,1), stride=2,padding=(1,0),output_padding=(1,0)),  # 输入通道数为128，输出通道数为64

        )
        self.xencoder = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=(1,3), stride=1,padding=(0,1)),  # 输入通道数为1，输出通道数为16
            #nn.MaxPool2d(1,2),  # 输入通道数为16，输出通道数为8
            nn.ReLU(),
            # nn.Conv2d(16, 8, kernel_size=(1,3), stride=1,padding=(0,1)),  # 输入通道数为1，输出通道数为16
            # nn.MaxPool2d(1,2),  # 输入通道数为16，输出通道数为8
            # nn.ReLU(),
        )
        
        self.xdecoder = nn.Sequential(
            #nn.ConvTranspose2d(8, 16, kernel_size=(1,3), stride=1,padding=(0,1)),  # 输入通道数为128，输出通道数为64
            #nn.ConvTranspose2d(16, 64, kernel_size=(1,3), stride=2,padding=(0,1)),  # 输入通道数为128，输出通道数为64
            nn.Conv2d(8, 16, kernel_size=(1,3), stride=1,padding=(0,1)),  # 输入通道数为1，输出通道数为16

        )

        fill_up_weights(self.net1)
        self.net2 = nn.ConvTranspose2d(64,64, (9,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
        fill_up_weights(self.net2)
        self.net3 = nn.Conv2d(64,16, kernel_size=1, bias=True)
        conv_init(self.net3)
        self.net4 = nn.Conv2d(64,16, kernel_size=1, bias=True)
        conv_init(self.net4)
        self.net5 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net5)
        self.net6 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net6)
        self.net7 = nn.Sequential(
				nn.Linear(832,64),#200,64
				#nn.Dropout(0.2),
				nn.ReLU(),
				nn.Linear(64,8))#8
        self.net8 = nn.Sequential(
				nn.Linear(417,64),#1,64
				#nn.Dropout(0.1),
				nn.ReLU(),
				nn.Linear(64,8))

        self.x_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(104,1), bias=True))#25,53
        self.y_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(1,53), bias=True))

    def forward(self, x):
        encoder = self.base(x) #resnet-18,即featuremap

        axis_y = self.y_pre(encoder) #Size compression
        axis_y = self.net1(axis_y)    #transpose convolution

        
        axis_y_branch = self.yencoder(axis_y)
        axis_y_branch = self.ydecoder(axis_y_branch)
        axis_y = self.net3(axis_y)   #1x1 convolution
        #axis_y = axis_y_branch +axis_y#加分支上的可学习卷积层
        axis_y = axis_y
        axis_y = nn.ReLU()(axis_y) 
        axis_y = self.net5(axis_y)#ytezheng 
        axis_y = nn.ReLU()(axis_y)

        x=axis_y.view(axis_y.shape[0],-1)

        coord_y = self.net7(x)

        axis_x = self.x_pre(encoder)
        axis_x = self.net2(axis_x)



        axis_x_branch= self.xencoder(axis_x)
        axis_x_branch = self.xdecoder(axis_x_branch)

        axis_x = self.net4(axis_x)
        #axis_x = axis_x + axis_x_branch#加分支上的可学习卷积层
        axis_x = nn.ReLU()(axis_x) 
        axis_x = self.net6(axis_x)#xtezheng canshujicheng
        axis_x = nn.ReLU()(axis_x)

        coord_x = self.net8(axis_x.view(axis_x.shape[0],-1))


        output = torch.cat((coord_x.view(-1,8,1),coord_y.view(-1,8,1)), 2)

        return output

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

class Net_backbonebranch(nn.Module):
    def __init__(self, classes, embed_dim, resnet, pretrained_model=None, pretrained=None, use_torch_up=False):
        super().__init__()
        assert(isinstance(classes , dict)), f"num_labels should be dict, got {type(classes)}"
      

        self.datasets = list(classes.keys())
        self.embed_dim = embed_dim

        resnet_archs = {'resnet_18':drn_d_22 , 'resnet_34':drn_d_38, 'resnet_50':drn_d_54 , 'resnet_101':drn_d_105}
        arch = resnet_archs[resnet]
        
        model = arch(pretrained=pretrained, num_classes=1000)
        #pmodel = nn.DataParallel(model)
        #if pretrained_model is not None:
        #	pmodel.load_state_dict(pretrained_model)

        self.base = nn.Sequential(*list(model.children())[:-2]) ## Encoder. 
        self.seg = nn.ModuleList() ## Decoder 1d conv
        self.up = nn.ModuleList() ## Decoder upsample (non-trainable)
        

        # self.resnet_branch = nn.Sequential(
		# 		nn.Conv2d(3,32, kernel_size = (5,5), padding = 2, stride = 1, bias = True),
		# 		nn.MaxPool2d(2,2),
        #         nn.BatchNorm2d(32),
		# 		nn.ReLU(),
        #         nn.Conv2d(32,64, kernel_size = (5,5),padding = 3,stride = 1, bias=True),
		# 		nn.MaxPool2d(2,2),
        #         nn.BatchNorm2d(64),
		# 		nn.ReLU(),
		# 		nn.Conv2d(64,64, kernel_size = (3,3),padding = (1,2),stride = 1, bias=True),
		# 		nn.MaxPool2d(2,2),
        #         nn.BatchNorm2d(64),
		# 		nn.ReLU(),
        #         nn.Conv2d(64,64, kernel_size = (3,3),padding = (1,2),stride = 1, bias=True),
		# 		nn.MaxPool2d(2,2),
        #         nn.BatchNorm2d(64),
		# 		nn.ReLU(),
        #         nn.ConvTranspose2d(64,64,(3,3),stride=2,padding=(1,1),output_padding=(1,0))
        #         )

        branchmodel = drn_d_14(pretrained=pretrained, num_classes=64)
        self.resnet_branch = nn.Sequential(*list(branchmodel.children())[:-2])
        #self.resnet_branch = nn.Sequential(*list(branchmodel.children())[:])


        # 遍历self.resnet_branch中的所有模块
        for module in self.resnet_branch.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # 对Conv2d和Linear层进行权重初始化
                torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    # 如果存在偏置项，也进行初始化
                    torch.nn.init.constant_(module.bias, 0)

        self.net1 = nn.ConvTranspose2d(64,64, (16,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
        fill_up_weights(self.net1)
        self.net2 = nn.ConvTranspose2d(64,64, (9,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
        fill_up_weights(self.net2)
        self.net3 = nn.Conv2d(64,16, kernel_size=1, bias=True)
        conv_init(self.net3)
        self.net4 = nn.Conv2d(64,16, kernel_size=1, bias=True)
        conv_init(self.net4)
        self.net5 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net5)
        self.net6 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net6)
        self.net7 = nn.Sequential(
				nn.Linear(832,64),#832,64
				#nn.Dropout(0.2),
				nn.ReLU(),
				nn.Linear(64,8))#8
        self.net8 = nn.Sequential(
				nn.Linear(417,64),#417,64
				#nn.Dropout(0.1),
				nn.ReLU(),
				nn.Linear(64,8))
        '''self.net7 = nn.ModuleDict([[ d , nn.Sequential(
				nn.Linear(200,64),
				#nn.Dropout(0.2),
				nn.ReLU(),
				nn.Linear(64,4))] for d in self.datasets])
		self.net8 = nn.ModuleDict([[ d , nn.Sequential(
				nn.Linear(417,64),
				#nn.Dropout(0.1),
				nn.ReLU(),
				nn.Linear(64,4))] for d in self.datasets])'''
        '''self.net7 = nn.Linear(830,64)
		self.net8 = nn.Linear(417,64)
		self.net9 = nn.Linear(64,4)
		self.net10 = nn.Linear(64,4)'''
        #self.x_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(25,53), bias=True))#104,1
        self.x_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(104,1), bias=True))#104,1
        self.y_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(1,53), bias=True))

    def forward(self, x):
        encoder_branch = self.resnet_branch(x)
        encoder1 = self.base(x)
        encoder = 0.5*encoder_branch + encoder1 #resnet-18+迁移网络
        axis_y = self.y_pre(encoder) #Size compression
        axis_y = self.net1(axis_y)    #transpose convolution

        

        axis_y = self.net3(axis_y)   #1x1 convolution
        axis_y = nn.ReLU()(axis_y)
        axis_y = self.net5(axis_y)
        axis_y = nn.ReLU()(axis_y)
        '''axis_y = self.net7(axis_y.view(axis_y.shape[0],-1)) 
        axis_y = nn.ReLU()(axis_y)
        coord_y = self.net9(axis_y)'''
        #coord_y = { d : self.net7[d](axis_y.view(axis_y.shape[0],-1)) for d in self.datasets}
        x=axis_y.view(axis_y.shape[0],-1)
        #coord_y = self.net7(axis_y.view(axis_y.shape[0],-1))
        coord_y = self.net7(x)

        axis_x = self.x_pre(encoder)
        axis_x = self.net2(axis_x)



        axis_x = self.net4(axis_x)
        axis_x = nn.ReLU()(axis_x)
        axis_x = self.net6(axis_x)
        axis_x = nn.ReLU()(axis_x)
        '''axis_x = self.net8(axis_x.view(axis_x.shape[0],-1)) 
        axis_x = nn.ReLU()(axis_x)
        coord_x = self.net10(axis_x)'''
        #coord_x = { d : self.net8[d](axis_x.view(axis_x.shape[0],-1)) for d in self.datasets}
        z = axis_x.view(axis_x.shape[0],-1)
        coord_x = self.net8(z)

        #output = {d : torch.cat((coord_x[d].view(-1,4,1),coord_y[d].view(-1,4,1)), 2) for d in self.datasets}
        output = torch.cat((coord_x.view(-1,8,1),coord_y.view(-1,8,1)), 2)

        return output

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

class Fire(nn.Module):#fire module
    def __init__(self, in_channels, s1x1_channels, e1x1_channels, e3x3_channels):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, s1x1_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(s1x1_channels, e1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(s1x1_channels, e3x3_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.squeeze(x))
        out1x1 = torch.relu(self.expand1x1(x))
        out3x3 = torch.relu(self.expand3x3(x))
        return torch.cat([out1x1, out3x3], 1)

class SqueezeNet(nn.Module):#并联backbone迁移网络（大参数量）
    def __init__(self, num_classes=64):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256)
        )
        self.final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((104, 53))  # 根据输出尺寸调整

    def forward(self, x):
        x = self.features(x)
        x = self.final_conv(x)
        x = self.avg_pool(x)
        return x

class SqueezeNet_small(nn.Module):#并联backbone迁移网络（小参数量）
    def __init__(self, num_classes=64):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(48, 8, 32, 32),
            Fire(64, 8, 32, 32),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 24, 96, 96),
            Fire(192, 24, 96, 96),
            Fire(192, 32, 128, 128),
            Fire(256, 32, 128, 128)
        )
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((104, 53))  # 根据输出尺寸调整

    def forward(self, x):
        x = self.features(x)
        x = self.final_conv(x)
        x = self.avg_pool(x)
        return x
    

class Net_ps_Transfer(nn.Module):#串联加并联迁移网络
    def __init__(self, classes, embed_dim, resnet, pretrained_model=None, pretrained=None, use_torch_up=False):
        super().__init__()
        assert(isinstance(classes , dict)), f"num_labels should be dict, got {type(classes)}"
        self.datasets = list(classes.keys())
        self.embed_dim = embed_dim

        resnet_archs = {'resnet_18':drn_d_22 , 'resnet_34':drn_d_38, 'resnet_50':drn_d_54 , 'resnet_101':drn_d_105}
        arch = resnet_archs[resnet]
        
        model = arch(pretrained=pretrained, num_classes=1000)
        #pmodel = nn.DataParallel(model)
        #if pretrained_model is not None:
        #	pmodel.load_state_dict(pretrained_model)

        self.base = nn.Sequential(*list(model.children())[:-2]) ## Encoder. 
        self.Squeezebase = SqueezeNet()#实例化squeezenet并联迁移网络
        self.channel_adjustment = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(192, 64, kernel_size=1))
        self.seg = nn.ModuleList() ## Decoder 1d conv
        self.up = nn.ModuleList() ## Decoder upsample (non-trainable)

        self.net1 = nn.ConvTranspose2d(64,64, (16,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
        # Downsample 1
        self.yencoder1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # 输入通道数为64，输出通道数为32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 输入尺寸为(4, 64, 832, 1)，输出尺寸为(4, 32, 416, 1)
        )
        # Downsample 2
        self.yencoder2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # 输入通道数为32，输出通道数为16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 输入尺寸为(4, 32, 416, 1)，输出尺寸为(4, 16, 208, 1)
        )
        # Downsample 3
        self.yencoder3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=(3, 1), stride=1, padding=(1, 0)),  # 输入通道数为16，输出通道数为8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 输入尺寸为(4, 16, 208, 1)，输出尺寸为(4, 8, 104, 1)
        )
        # Upsample 1
        self.ydecoder1 = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), output_padding=(1, 0)),  # 输入尺寸为(4, 8, 104, 1)，输出尺寸为(4, 16, 208, 1)
        )
        # Upsample 2
        self.ydecoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), output_padding=(1, 0)),  # 输入尺寸为(4, 16, 208, 1)，输出尺寸为(4, 32, 416, 1)
        )
        # Upsample 3
        self.ydecoder3 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=(2, 1), stride=(2, 1)),  # 输入尺寸为(4, 32, 416, 1)，输出尺寸为(4, 64, 832, 1)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),  # 输入通道数为64，输出通道数为64
        )
        self.xencoder1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 3), stride=1, padding=(0, 1)),  # Input: (40, 64, 1, 417), Output: (40, 32, 1, 417)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Input: (40, 32, 1, 417), Output: (40, 32, 1, 208)
        )
        # Downsample 2
        self.xencoder2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(1, 3), stride=1, padding=(0, 1)),  # Input: (40, 32, 1, 208), Output: (40, 16, 1, 208)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),   # Input: (40, 16, 1, 208), Output: (40, 16, 1, 104)
        )
        # Downsample 3
        self.xencoder3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=(1, 3), stride=1, padding=(0, 1)),  # Input: (40, 16, 1, 104), Output: (40, 8, 1, 104)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),   # Input: (40, 8, 1, 104), Output: (40, 8, 1, 52)
        )
        # Upsample 1
        self.xdecoder1 = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=(1, 2), stride=2),  # Input: (40, 8, 1, 52), Output: (40, 16, 1, 104)
        )
        self.xdecoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=(1, 3), stride=2, padding=(0, 1), output_padding=(0, 1)),  # Input: (40, 16, 1, 104), Output: (40, 32, 1, 208)
        )
        # Upsample 3
        self.xdecoder3 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=(1, 4), stride=2, padding=(0, 1), output_padding=(0, 1)), 
        )
        fill_up_weights(self.net1)
        self.net2 = nn.ConvTranspose2d(64,64, (9,9), stride=8, padding=4,
										output_padding=0, groups=1,
										bias=False)
        fill_up_weights(self.net2)
        self.net3 = nn.Conv2d(64,16, kernel_size=1, bias=True)
        conv_init(self.net3)
        self.net4 = nn.Conv2d(64,16, kernel_size=1, bias=True)
        conv_init(self.net4)
        self.net5 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net5)
        self.net6 = nn.Conv2d(16,1, kernel_size=1, bias=True)
        conv_init(self.net6)
        self.net7 = nn.Sequential(
				nn.Linear(832,64),#200,64
				#nn.Dropout(0.2),
				nn.ReLU(),
				nn.Linear(64,8))#8
        self.net8 = nn.Sequential(
				nn.Linear(417,64),#1,64
				#nn.Dropout(0.1),
				nn.ReLU(),
				nn.Linear(64,8))
        '''self.net7 = nn.ModuleDict([[ d , nn.Sequential(
				nn.Linear(200,64),
				#nn.Dropout(0.2),
				nn.ReLU(),
				nn.Linear(64,4))] for d in self.datasets])
		self.net8 = nn.ModuleDict([[ d , nn.Sequential(
				nn.Linear(417,64),
				#nn.Dropout(0.1),
				nn.ReLU(),
				nn.Linear(64,4))] for d in self.datasets])'''
        '''self.net7 = nn.Linear(830,64)
		self.net8 = nn.Linear(417,64)
		self.net9 = nn.Linear(64,4)
		self.net10 = nn.Linear(64,4)'''
        self.x_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(104,1), bias=True))#25,53
        self.y_pre = nn.Sequential(nn.Conv2d(model.out_dim,64, kernel_size=(1,53), bias=True))

    def forward(self, x):
        encoder_branch = self.Squeezebase(x)#并联迁移微调部分
        
        encoder = self.base(x)#resnet-18,即featuremap
        encoder = encoder+encoder_branch*1.5
        axis_y = self.y_pre(encoder) #Size compression
        axis_y = self.net1(axis_y)    #transpose convolution
        axis_y = self.yencoder1(axis_y)#串联迁移微调部分
        axis_y = self.yencoder2(axis_y)
        axis_y = self.yencoder3(axis_y)
        axis_y = self.ydecoder1(axis_y)
        axis_y = self.ydecoder2(axis_y)
        axis_y = self.ydecoder3(axis_y)
        axis_y = self.net3(axis_y)   #1x1 convolution
        axis_y = nn.ReLU()(axis_y)
        axis_y = self.net5(axis_y)
        axis_y = nn.ReLU()(axis_y)
        '''axis_y = self.net7(axis_y.view(axis_y.shape[0],-1)) 
        axis_y = nn.ReLU()(axis_y)
        coord_y = self.net9(axis_y)'''
        #coord_y = { d : self.net7[d](axis_y.view(axis_y.shape[0],-1)) for d in self.datasets}
        z=axis_y.view(axis_y.shape[0],-1)
        #coord_y = self.net7(axis_y.view(axis_y.shape[0],-1))
        coord_y = self.net7(z)

        axis_x = self.x_pre(encoder)
        axis_x = self.net2(axis_x)
        axis_x = self.xencoder1(axis_x)
        axis_x = self.xencoder2(axis_x)
        axis_x = self.xencoder3(axis_x)
        axis_x = self.xdecoder1(axis_x)
        axis_x = self.xdecoder2(axis_x)
        axis_x = self.xdecoder3(axis_x)
        axis_x = self.net4(axis_x)
        axis_x = nn.ReLU()(axis_x)
        axis_x = self.net6(axis_x)#xtezheng canshujicheng
        axis_x = nn.ReLU()(axis_x)
        '''axis_x = self.net8(axis_x.view(axis_x.shape[0],-1)) 
        axis_x = nn.ReLU()(axis_x)
        coord_x = self.net10(axis_x)'''
        #coord_x = { d : self.net8[d](axis_x.view(axis_x.shape[0],-1)) for d in self.datasets}
        coord_x = self.net8(axis_x.view(axis_x.shape[0],-1))

        #output = {d : torch.cat((coord_x[d].view(-1,4,1),coord_y[d].view(-1,4,1)), 2) for d in self.datasets}
        output = torch.cat((coord_x.view(-1,8,1),coord_y.view(-1,8,1)), 2)

        return output

    def optim_parameters(self, memo=None):

        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param
        for param in self.Squeezebase.parameters():
            yield param

if __name__ == '__main__':
    model = drn_d_54(pretrained=False)
    input_tensor = torch.randn(1, 3, 224, 224)  # 创建一个随机输入张量
    output = model(input_tensor)  # 前向传播，打印出添加adapter的位置
