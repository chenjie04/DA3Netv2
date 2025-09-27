"""
Code for Traffic-sign recognition.

"""

from ast import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

from ultralytics.nn.modules.cbam import CBAM

class SCDown_v2(nn.Module):

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """
        Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=True)
        self.cv3 = Conv(c2, c2, 1, 1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv3(self.cv2(self.cv1(x)))


class DualAxisAggAttn_v3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 4,
        middle_ratio: float = 0.5,
    ):
        super().__init__()
        self.channels = in_channels
        self.groups = groups
        middle_channels = int(in_channels * middle_ratio)
        self.middle_channels = middle_channels

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.qkv = nn.ModuleDict(
            {
                "W": nn.Conv2d(
                    in_channels=middle_channels,
                    out_channels=middle_channels * 3,
                    kernel_size=1,
                    groups=groups,
                ),
                "H": nn.Conv2d(
                    in_channels=middle_channels,
                    out_channels=middle_channels * 3,
                    kernel_size=1,
                    groups=groups,
                ),
            }
        )

        self.conv_fusion = nn.ModuleDict(
            {
                "W": Conv(
                    c1=middle_channels,
                    c2=middle_channels,
                    k=3,
                    g=middle_channels,
                    act=True,
                ),
                "H": Conv(
                    c1=middle_channels,
                    c2=middle_channels,
                    k=3,
                    g=middle_channels,
                    act=True,
                ),
            }
        )

        final_channels = int(2 * middle_channels)
        self.out_project = Conv(c1=final_channels, c2=out_channels, k=1, act=True)

    def _apply_axis_attention(self, x, axis):
        """通用轴注意力计算"""
        qkv = self.qkv[axis](x)
        query, key, value = torch.split(
            qkv,
            [self.middle_channels] * 3,
            dim=1,
        )

        dim = -1 if axis == "W" else -2
        query_avg = query.mean(dim=dim, keepdim=True)
        scores = F.softmax(query_avg * key, dim=dim)
        context = (value * scores).sum(dim=dim, keepdim=True)

        gate = F.sigmoid(x)
        out = x + gate * context.expand_as(x)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, H, W]

        Returns:
            torch.Tensor: 输出张量，形状为 [B, C, H, W]
        """
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)

        # 宽轴注意力
        x_W = self._apply_axis_attention(x_main, "W")
        x_W_fused = self.conv_fusion["W"](x_W) + x_W
        # 高轴注意力
        x_H = self._apply_axis_attention(x_W_fused, "H")
        x_H_fused = self.conv_fusion["H"](x_H) + x_H

        x_out = torch.cat([x_H_fused, x_short], dim=1)

        x_out = self.out_project(x_out)

        return x_out


def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert num_channels % groups == 0, "num_channels should be " "divisible by groups"
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class LocalExtractor(nn.Module):
    def __init__(self, channels: int, experts: int = 2, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.experts = experts
        self.middle_channels = channels // 2

        # Step 1: 降维
        self.intrinsic_conv = Conv(
            channels, self.middle_channels, kernel_size, act=True
        )

        # Step 2: depthwise-like 空间建模
        self.spatial_conv = Conv(
            self.middle_channels,
            self.middle_channels * experts * 2,  # x2 是为了计算GLU
            kernel_size,
            g=self.middle_channels,
            act=True,
        )

        # Step 3: 跨通道融合
        self.reduce = Conv(
            self.middle_channels * experts * 2,
            self.middle_channels * 2,
            k=1,
            g=2,
            act=True,
        )  # 2x channels

        self.project = Conv(channels, channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intrinsic_feat = self.intrinsic_conv(x)
        ghost_feat = self.spatial_conv(intrinsic_feat)

        ghost_feat, gate = self.reduce(ghost_feat).chunk(2, dim=1)
        gate = F.sigmoid(gate)
        ghost_feat = gate * ghost_feat

        out = torch.cat([ghost_feat, intrinsic_feat], dim=1)
        out = channel_shuffle(out, 2)
        out = self.project(out)

        return out


class ELANBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_experts: int = 4,
        middle_ratio: float = 0.5,
        num_blocks: int = 2,
    ):
        super().__init__()

        middle_channels = int(in_channels * middle_ratio)
        final_channels = int((2 + num_blocks) * middle_channels)

        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            internal_block = LocalExtractor(
                channels=middle_channels, kernel_size=kernel_size
            )
            self.blocks.append(internal_block)

        self.out_project = Conv(c1=final_channels, c2=out_channels, k=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        block_outs = []
        x_block = x_main
        for block in self.blocks:
            x_block = block(x_block)
            block_outs.append(x_block)
        x_final = torch.cat((*block_outs[::-1], x_main, x_short), dim=1)
        return self.out_project(x_final)


class TSRModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attn_groups: int = 4,
    ) -> None:
        super().__init__()

        self.attn = DualAxisAggAttn_v3(
            in_channels=in_channels, out_channels=out_channels, groups=attn_groups
        )
        self.local_extractor = ELANBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.attn(x) + residual
        residual = x
        x = self.local_extractor(x) + residual

        return x


# class DySA_ReLU(nn.Module):
#     """
#     Dynamic Spatial-Channel Adaptive ReLU: Dynamic ReLU with Spatial and Channel-wise Attention
#     """

#     def __init__(
#         self,
#         channels: int,
#         reduction: int = 4,
#     ):
#         super(DySA_ReLU, self).__init__()

#         self.channels = channels
#         self.expansion = 2  # for a, b

#         # a1, b1
#         self.channel_attn_avg = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction, 1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, channels * self.expansion, 1, bias=True),
#             nn.Hardsigmoid(),
#         )
#         self.spatial_attn_avg = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
#             nn.Hardsigmoid(),
#         )

#         # a2, b2
#         self.channel_attn_max = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction, 1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, channels * self.expansion, 1, bias=True),
#             nn.Hardsigmoid(),
#         )
#         self.spatial_attn_max = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
#             nn.Hardsigmoid(),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         channel_avg_pool = F.avg_pool2d(
#             x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
#         )
#         channel_att_avg = (
#             self.channel_attn_avg(channel_avg_pool) - 0.5
#         )  # value range [-0.5, 0.5]
#         a1, b1 = torch.split(channel_att_avg, self.channels, dim=1)
#         a1 = a1 * 2.0 + 1.0  # value range # [-1.0, 1.0] + 1.0
#         spatial_avg = torch.mean(x, dim=(1), keepdim=True)
#         spatial_attn_avg = self.spatial_attn_avg(spatial_avg)
#         a1 = a1 * spatial_attn_avg

#         channel_max_pool = F.max_pool2d(
#             x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
#         )
#         channel_att_max = self.channel_attn_max(channel_max_pool) - 0.5
#         a2, b2 = torch.split(channel_att_max, self.channels, dim=1)
#         a2 = a2 * 2.0  # value range [-1.0, 1.0]
#         spatial_max = torch.max(x, dim=(1), keepdim=True)[0]
#         spatial_attn_max = self.spatial_attn_max(spatial_max)
#         a2 = a2 * spatial_attn_max

#         out = torch.max(x * a1 + b1, x * a2 + b2)

#         return out + x

# v1.0 有效果
class AdaConcat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension in a adaptive manner.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(
        self, channels: list = [512, 256], reduction: int = 4, dimension: int = 1
    ):
        """
        Initialize Concat module.

        Args:
            channels (List[int]): List of input channels.
            reduction (int): Reduction ratio for channel attention.
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.d = dimension


        total_chs = sum(self.channels)
        self.channel_attn_max = nn.Sequential(
            nn.Conv2d(total_chs, total_chs // self.reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_chs // self.reduction, total_chs, 1, bias=True),
            nn.Hardsigmoid(),
        )

        self.spatial_attn_max = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=True),
            nn.Hardsigmoid(),
        )

    def forward(self, x: list[torch.Tensor]):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        channel_max_pool = torch.cat(
            [
                F.max_pool2d(
                    x[i],
                    (x[i].size(2), x[i].size(3)),
                    stride=(x[i].size(2), x[i].size(3)),
                )
                for i in range(len(x))
            ],
            dim=1,
        )

        channel_att_max = self.channel_attn_max(channel_max_pool)
        a1, a2 = torch.split(channel_att_max, self.channels, dim=1)

        spatial_max = torch.cat(
            [
                torch.max(x[i], dim=(1), keepdim=True)[0]
                for i in range(len(x))
            ],
            dim=1,
        )
        
        spatial_att_max = self.spatial_attn_max(spatial_max)
        s1, s2 = torch.split(spatial_att_max, [1,1], dim=1)

        out = torch.cat([x[0] * torch.sigmoid(a1*s1), x[1] * torch.sigmoid(a2*s2)], dim=self.d)

        return out