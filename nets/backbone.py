# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from .ops import NestedTensor, is_main_process
from nets.attention import cbam_block, eca_block, se_block, CA_Block
from torchvision.models.resnet import resnet50

attention_block = [se_block, cbam_block, eca_block, CA_Block]
# ----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------#
#   卷积块
#   Conv2d + BatchNorm2d + LeakyReLU
# -------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        #self.bn = nn.BatchNorm2d(out_channels)
        self.bn = FrozenBatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

'''
                    input
                      |
                  BasicConv
                      -----------------------
                      |                     |
                 route_group              route
                      |                     |
                  BasicConv                 |
                      |                     |
    -------------------                     |
    |                 |                     |
 route_1          BasicConv                 |
    |                 |                     |
    -----------------cat                    |
                      |                     |
        ----      BasicConv                 |
        |             |                     |
      feat           cat---------------------
                      |
                 MaxPooling2D
'''


# ---------------------------------------------------#
#   CSPdarknet53-tiny的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
# ---------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels

        self.conv1 = BasicConv(in_channels, out_channels, 3)

        self.conv2 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.conv3 = BasicConv(out_channels // 2, out_channels // 2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x

        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c // 2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)

        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)

        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x, feat


class CSPDarkNet(nn.Module):
    def __init__(self):
        super(CSPDarkNet, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 = Resblock_body(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 = Resblock_body(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 = Resblock_body(256, 256)
        # 13,13,512 -> 13,13,512
        self.conv3 = BasicConv(512, 512, kernel_size=3)

        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _ = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _ = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1 = self.resblock_body3(x)

        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2







def darknet53_tiny(pretrained, **kwargs):
    model = CSPDarkNet()
    if pretrained:
        model.load_state_dict(torch.load("model_data/CSPdarknet53_tiny_backbone_weights.pth"))
    return model


#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#--------------------------------------!!!PAN-----------------------------------------------------
#-------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + LeakyReLU
#-------------------------------------------------#
class BasicConv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            # scale_factor=2 表示将输入 tensor 的宽和高扩大两倍
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x




#-------------------------------------------------------------------------------------------------
class PositionEmbeddingSine(nn.Module):
    """
    这是一个更标准的位置嵌入版本，按照sine进行分布
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats  = num_pos_feats
        self.temperature    = temperature
        self.normalize      = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x           = tensor_list.tensors
        mask        = tensor_list.mask
        assert mask is not None
        not_mask    = ~mask
        y_embed     = not_mask.cumsum(1, dtype=torch.float32)
        x_embed     = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps     = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    创建可学习的位置向量
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x       = tensor_list.tensors
        h, w    = x.shape[-2:]
        i       = torch.arange(w, device=x.device)
        j       = torch.arange(h, device=x.device)
        x_emb   = self.col_embed(i)
        y_emb   = self.row_embed(j)
        pos     = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

def build_position_encoding(position_embedding, hidden_dim=256):
    # 创建位置向量
    N_steps = hidden_dim // 2
    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {position_embedding}")

    return position_embedding

class FrozenBatchNorm2d(torch.nn.Module):
    """
    冻结固定的BatchNorm2d。
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w       = self.weight.reshape(1, -1, 1, 1)
        b       = self.bias.reshape(1, -1, 1, 1)
        rv      = self.running_var.reshape(1, -1, 1, 1)
        rm      = self.running_mean.reshape(1, -1, 1, 1)
        eps     = 1e-5
        scale   = w * (rv + eps).rsqrt()
        bias    = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):
    """
    用于指定返回哪个层的输出
    这里返回的是最后一层
    """
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int,phi):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
            
        # 用于指定返回的层
        self.body           = backbone
        self.num_channels   = num_channels

        self.phi = phi
        # self.backbone = darknet53_tiny(pretrained)

        self.conv_for_P5 = BasicConv(512, 256, 1)
        # self.yolo_headP5 = yolo_head([512, len(anchors_mask[0]) * (5 + num_classes)], 256)

        self.upsample = Upsample(256, 128)
        # self.yolo_headP4 = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)], 384)

        if 1 <= self.phi and self.phi <= 4:
            self.feat1_att = attention_block[self.phi - 1](256)
            self.feat2_att = attention_block[self.phi - 1](512)
            self.upsample_att = attention_block[self.phi - 1](128)

    def forward(self, tensor_list: NestedTensor):
        feat1, feat2= self.body(tensor_list.tensors)
        if 1 <= self.phi and self.phi <= 4:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)

        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2)
        # 13,13,256 -> 13,13,512 -> 13,13,255
        # out0 = self.yolo_headP5(P5)

        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample(P5)
        # 26,26,256 + 26,26,128 -> 26,26,384
        if 1 <= self.phi and self.phi <= 4:
            P5_Upsample = self.upsample_att(P5_Upsample)
        P4 = torch.cat([P5_Upsample, feat1], axis=1)
        xs={'out':P4}

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m           = tensor_list.mask
            assert m is not None
            mask        = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name]   = NestedTensor(x, mask)
        return out

class Backbone(BackboneBase):
    """
    ResNet backbone with frozen BatchNorm.
    """
    def __init__(self,  train_backbone: bool,phi,pretrained:bool):
        # 首先利用torchvision里面的model创建一个backbone模型
        ''' backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation    = [False, False, dilation],
            pretrained                      = pretrained, 
            norm_layer                      = FrozenBatchNorm2d
        )'''
        #backbone =darknet53_tiny(pretrained=False)
        backbone=CSPDarkNet()
        if pretrained:
            backbone.load_state_dict(torch.load("model_data/CSPdarknet53_tiny_backbone_weights.pth"))

        # 根据选择的模型，获得通道数
        #num_channels =  2048
        num_channels = 384
        super().__init__(backbone, train_backbone, num_channels, phi)

class Joiner(nn.Sequential):
    """
    用于将主干和位置编码模块进行结合
    """
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs                      = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos                     = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

def build_backbone( position_embedding, hidden_dim, phi=0, train_backbone=True,pretrained=False ):
    # 创建可学习的位置向量还是固定的按'sine'排布的位置向量
    position_embedding  = build_position_encoding(position_embedding, hidden_dim)
    # 创建主干
    backbone            = Backbone( train_backbone,   phi,pretrained=pretrained )
    
    # 用于将主干和位置编码模块进行结合
    model               = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model