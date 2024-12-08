import math
import torch
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F
import numpy as np
def make_layer(basic_block, num_basic_block, **kwarg):#用于创建由相同基本块组成的层，并将这些基本块堆叠起来

    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
        **kwarg 允许传入额外的关键字参数，这些参数将传递给基本块的构造函数
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):#循环迭代几次
        layers.append(basic_block(**kwarg))#每次迭代中，通过basic_来构造函数，并传入**kwarg中任何额外参数，创建一个新的基本块实例，并添加到layers列表中
    return nn.Sequential(*layers)#将所有块堆叠在一起并返回。

@ARCH_REGISTRY.register()
class MDBN(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, upscale=2, res_scale=1.0):
        super(MDBN, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)






        #用于创建多个残差块，其中每个残差块由ResidualBlock类组成。num_block参数指定了创建的残差块数量，而num_feat和res_scale将参数传递给ResidualBlock类的初始化方法
        #self.body = make_layer(ResidualBlock, num_block, num_feat=num_feat, res_scale=res_scale)
        #这个卷积层将num_feat个特征的图像转换为另外num_feat个特征的推向，其卷积核大小为3*3，步长为1，填充为1
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_feat=num_feat, res_scale=res_scale) for _ in range(num_block)
        ])

        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)


        self.upsample = Upsample(upscale, num_feat)

        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x
        x = self.conv_last(self.upsample(res))


        return x

class ResidualBlock(nn.Module):#实现了残差块的功能
    def __init__(self, num_feat=64, res_scale=1):#特征数量64，res_scale残差比例，默认为1，残差缩放比例的作用是在残差网络中调节残差的重要性，提高模型的性能和训练稳定性。
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale#用于控制残差的缩放比例
        self.baseblock1 = BaseBlock(num_feat)#创建一个名为baseblock1的成员变量
        self.baseblock2 = BaseBlock(num_feat)




    def forward(self, x):
        identity = x#在后续残差中使用

        x = self.baseblock1(x)
        x = self.baseblock2(x)


        return identity + x * self.res_scale

        #return identity + x * self.res_scale







class BaseBlock(nn.Module):#构建基本的卷积块
    def __init__(self, num_feat):
        super(BaseBlock, self).__init__()
        self.uconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.uconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.dconv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.act = nn.GELU()

    def forward(self, x):


        x1 = self.uconv2(self.act(self.uconv1(x)))#x经过第一个卷积层，然后通过GELU激活函数act，最后经过第二个卷积层，赋值给x1
        x2 = self.dconv(x)#经dconv卷积层，赋值给x2
        x = self.act(x1 + x2)



        return x

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)



class NRES(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6, stride=1):
        super(NRES, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expansion_factor

        layers = []
        if expansion_factor != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class LHSB(nn.Module):
    def __init__(self,
                 dim,
                 attn_drop=0.,
                 proj_drop=0.,
                 n_levels=4, ):

        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList([
            downsample_vit(dim // 4,
                           window_size=8,
                           attn_drop=attn_drop,
                           proj_drop=proj_drop,
                           down_scale=2 ** i)
            for i in range(self.n_levels)])

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        SA_before_idx = None
        out = []

        downsampled_feat = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                downsampled_feat.append(s)

            else:
                downsampled_feat.append(xc[i])

        for i in reversed(range(self.n_levels)):
            s = self.mfr[i](downsampled_feat[i])
            s_upsample = F.interpolate(s, size=(s.shape[2] * 2, s.shape[3] * 2), mode='nearest')

            if i > 0:
                downsampled_feat[i - 1] = downsampled_feat[i - 1] + s_upsample

            s_original_shape = F.interpolate(s, size=(h, w), mode='nearest')
            out.append(s_original_shape)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


class downsample_vit(nn.Module):
    def __init__(self,
                 dim,
                 window_size=8,
                 attn_drop=0.,
                 proj_drop=0.,
                 down_scale=2, ):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    def window_reverse(self, windows, window_size, h, w):
        """
        Args:
            windows: (num_windows*b, window_size, window_size, c)
            window_size (int): Window size
            h (int): Height of image
            w (int): Width of image

        Returns:
            x: (b, h, w, c)
        """
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.window_size, self.window_size
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()

        x = x.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()
        return x, lepe

    def forward(self, x):
        B, C, H, W = x.shape

        ################################
        # 1. window partition
        ################################
        x = x.permute(0, 2, 3, 1)
        x_window = self.window_partition(x, self.window_size).permute(0, 3, 1, 2)
        x_window = x_window.permute(0, 2, 3, 1).view(-1, self.window_size * self.window_size, C)

        ################################
        # 2. make qkv
        ################################
        qkv = self.qkv(x_window)
        # qkv = qkv.permute(0,2,3,1)
        # qkv = qkv.reshape(-1, self.window_size * self.window_size, 3*C)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        ################################
        # 3. attn and PE
        ################################
        v, lepe = self.get_lepe(v, self.get_v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        # x = x.reshape(-1, self.window_size, self.window_size, C)
        # x = x.permute(0,3,1,2)

        ################################
        # 4. proj and drop
        ################################
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.reshape(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(x, self.window_size, H, W)

        return x.permute(0, 3, 1, 2)
