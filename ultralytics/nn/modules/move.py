
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.torch_utils import model_info
# from ultralytics.nn.modules.criss_cross_attn import CrissCrossAttention

import logging

logger = logging.getLogger(__name__)

USE_FLASH_ATTN = False
try:
    import torch

    if (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func

        USE_FLASH_ATTN = True
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa

        logger.warning(
            "FlashAttention is not available on this device. Using scaled_dot_product_attention instead."
        )
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    logger.warning(
        "FlashAttention is not available on this device. Using scaled_dot_product_attention instead."
    )


class SelfAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(  # type: ignore
                q.contiguous().half(), k.contiguous().half(), v.contiguous().half()
            ).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = v @ attn.transpose(-2, -1)

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class MoVE(nn.Module):
    """
    MoVE: Multi-experts Convolutional Neural Network.

    """

    def __init__(
        self,
        channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts

        assert channels % num_experts == 0, "channels must be divisible by num_experts"

        # 并行化专家计算
        self.experts = Conv(
            channels, channels * num_experts, kernel_size, g=channels, act=True
        )

        # 跨通道融合
        self.fusion = Conv(
                channels * num_experts,
                channels,
                k=1,
                g=num_experts,
                act=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        expert_outputs = self.experts(x)  # (B, C*A, H, W)

        out = self.fusion(expert_outputs)

        return out
    

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


class MoVE_GhostModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        main_kernel_size: int = 3,  # 主分支的卷积核大小
        num_experts: int = 16,  # 轻量分支专家数量
        cheap_kernel_size: int = 3,
    ):
        super().__init__()

        self.middle_channels = int(in_channels // 2)
        self.primary_conv = Conv(
            in_channels, self.middle_channels, k=main_kernel_size, act=True
        )

        self.cheap_operation = MoVE(
            self.middle_channels,
            num_experts,
            kernel_size=cheap_kernel_size,
        )

        self.out_project = Conv(self.middle_channels * 2, out_channels, k=1, act=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(out, 2)
        out = self.out_project(out)
        return out

class GhostModule(nn.Module):
    """ """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,  # 主分支的卷积核大小
    ):
        super().__init__()

        self.middle_channels = int(in_channels // 2)
        self.primary_conv = Conv(
            in_channels, self.middle_channels, k=kernel_size, act=True
        )

        self.cheap_operation = Conv(
            self.middle_channels,
            self.middle_channels,
            k=3,
            g=self.middle_channels,
            act=True,
        )  # 3x3深度分离卷积, 即num_experts=1

        # self.out_project = Conv(self.middle_channels * 2, out_channels, k=1, act=True)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        # out = channel_shuffle(out, 2)
        # out = self.out_project(out)
        return out

class MG_ELAN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_experts: int = 16,
        middle_ratio: float = 0.5,
        num_blocks: int = 2,
    ):
        super().__init__()

        middle_channels = int(in_channels * middle_ratio)
        block_channels = int(in_channels * middle_ratio)
        final_channels = int(2 * middle_channels) + int(num_blocks * block_channels)
        self.main_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=middle_channels, k=1, act=True)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            internal_block = MoVE_GhostModule(
                in_channels=middle_channels,
                out_channels=block_channels,
                main_kernel_size=kernel_size,
                num_experts=num_experts,
                cheap_kernel_size=3,
            )
            # internal_block = GhostModule(
            #     in_channels=middle_channels,
            #     out_channels=block_channels,
            #     kernel_size=1,
            # )
            middle_channels = block_channels
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

class light_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            Conv(c1=in_channels, c2=in_channels, k=1, act=True),
            Conv(c1=in_channels, c2=in_channels, k=3, g=in_channels, act=True),
            Conv(c1=in_channels, c2=out_channels, k=1, act=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)

class DualAxisAggAttn(nn.Module):
    def __init__(
        self,
        channels: int,
        middle_ratio: float = 0.5,
    ):
        super().__init__()
        self.channels = channels
        middle_channels = int(channels * middle_ratio)
        self.middle_channels = middle_channels

        self.main_conv = Conv(c1=channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=channels, c2=middle_channels, k=1, act=True)

        self.qkv = nn.ModuleDict(
            {
                "W": nn.Conv2d(
                    in_channels=middle_channels,
                    out_channels=1 + 2 * middle_channels,
                    kernel_size=1,
                    bias=True,
                ),
                "H": nn.Conv2d(
                    in_channels=middle_channels,
                    out_channels=1 + 2 * middle_channels,
                    kernel_size=1,
                    bias=True,
                ),
            }
        )

        self.conv_fusion = nn.ModuleDict(
            {
                "W": light_ConvBlock(
                    in_channels=middle_channels, out_channels=middle_channels
                ),
                "H": light_ConvBlock(
                    in_channels=middle_channels, out_channels=middle_channels
                ),
            }
        )


        final_channels = int(2 * middle_channels)
        self.out_project = Conv(c1=final_channels, c2=channels, k=1, act=True)

    def _apply_axis_attention(self, x, axis):
        """通用轴注意力计算"""
        qkv = self.qkv[axis](x)
        query, key, value = torch.split(
            qkv, [1, self.middle_channels, self.middle_channels], dim=1
        )

        # 明确指定softmax维度
        dim = -1 if axis == "W" else -2
        context_scores = F.softmax(query, dim=dim)
        context_vector = (key * context_scores).sum(dim=dim, keepdim=True)
        # gate = F.tanh(self.alpha[axis] * value) # 效果不及sigmoid
        # gate = F.silu(value) # 效果最差
        gate = F.sigmoid(value)
        # 将全局上下文向量乘以权重，并广播注入到特征图中
        out = x + gate * context_vector.expand_as(value)
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



class DualAxisAggAttn_no_aggregation(nn.Module):
    def __init__(
        self,
        channels: int,
        middle_ratio: float = 0.5,
    ):
        super().__init__()
        self.channels = channels
        middle_channels = int(channels * middle_ratio)
        self.middle_channels = middle_channels

        self.main_conv = Conv(c1=channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=channels, c2=middle_channels, k=1, act=True)


        self.conv_fusion = nn.ModuleDict(
            {
                "W": light_ConvBlock(
                    in_channels=middle_channels, out_channels=middle_channels
                ),
                "H": light_ConvBlock(
                    in_channels=middle_channels, out_channels=middle_channels
                ),
            }
        )


        final_channels = int(2 * middle_channels)
        self.out_project = Conv(c1=final_channels, c2=channels, k=1, act=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, H, W]

        Returns:
            torch.Tensor: 输出张量，形状为 [B, C, H, W]
        """
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
 
        x_W = self.conv_fusion["W"](x_main) + x_main
 
        x_H = self.conv_fusion["H"](x_W) + x_W

        x_out = torch.cat([x_H, x_short], dim=1)

        x_out = self.out_project(x_out)

        return x_out

class DualAxisAggAttn_no_fusion(nn.Module):
    def __init__(
        self,
        channels: int,
        middle_ratio: float = 0.5,
    ):
        super().__init__()
        self.channels = channels
        middle_channels = int(channels * middle_ratio)
        self.middle_channels = middle_channels

        self.main_conv = Conv(c1=channels, c2=middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=channels, c2=middle_channels, k=1, act=True)

        self.qkv = nn.ModuleDict(
            {
                "W": nn.Conv2d(
                    in_channels=middle_channels,
                    out_channels=1 + 2 * middle_channels,
                    kernel_size=1,
                    bias=True,
                ),
                "H": nn.Conv2d(
                    in_channels=middle_channels,
                    out_channels=1 + 2 * middle_channels,
                    kernel_size=1,
                    bias=True,
                ),
            }
        )

        final_channels = int(2 * middle_channels)
        self.out_project = Conv(c1=final_channels, c2=channels, k=1, act=True)

    def _apply_axis_attention(self, x, axis):
        """通用轴注意力计算"""
        qkv = self.qkv[axis](x)
        query, key, value = torch.split(
            qkv, [1, self.middle_channels, self.middle_channels], dim=1
        )

        # 明确指定softmax维度
        dim = -1 if axis == "W" else -2
        context_scores = F.softmax(query, dim=dim)
        context_vector = (key * context_scores).sum(dim=dim, keepdim=True)
        # gate = F.tanh(self.alpha[axis] * value) # 效果不及sigmoid
        # gate = F.silu(value) # 效果最差
        gate = F.sigmoid(value)
        # 将全局上下文向量乘以权重，并广播注入到特征图中
        out = x + gate * context_vector.expand_as(value)
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

        # 高轴注意力
        x_H = self._apply_axis_attention(x_W, "H")


        x_out = torch.cat([x_H, x_short], dim=1)

        x_out = self.out_project(x_out)

        return x_out

class TransMoVE(nn.Module):
    """采用ELAN结构"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
    ):
        super().__init__()
        # 注意力子层
        # --------------------------------------------------------------

        self.attn = DualAxisAggAttn(channels=in_channels)
        # self.attn = DualAxisAggAttn_no_aggregation(channels=in_channels)
        # self.attn = DualAxisAggAttn_no_fusion(channels=in_channels)

        # self attention
        # ---------------------------------------------------------------
        # summary: 272 layers, 2,618,176 parameters, 2,618,160 gradients, 6.9 GFLOPs
        # print(f"TransMoVE: {in_channels} -> {out_channels}")
        # num_heads = max(1, out_channels // 64)
        # self.attn = SelfAttn(dim=out_channels,num_heads=num_heads, area=1)

        # Area attention
        # ---------------------------------------------------------------
        # summary: 272 layers, 2,618,176 parameters, 2,618,160 gradients, 6.9 GFLOPs
        # num_heads = max(1, out_channels // 64)
        # area = int(512 /  out_channels)
        # if area < 4:
        #     area = 1
        # if out_channels == 256: # 256是n scale模型第4阶段的输入通道数，这里是一个取巧的做法
        #     area = 1
        # else:
        #     area = 4
        # # 主干网络的最后一个阶段不能划分，不然推理阶段由于图像分辨率不能整除而报错
        # self.attn = SelfAttn(dim=out_channels,num_heads=num_heads, area=area)

        # Criss Cross Attention
        # summary: 252 layers, 2,375,580 parameters, 2,375,564 gradients, 6.2 GFLOPs
        # ---------------------------------------------------
        # self.attn = CrissCrossAttention(in_channels)

        self.norm1 = nn.BatchNorm2d(in_channels)
        # 局部特征提取模块
        #  --------------------------------------------------------------
        self.local_extractor = MG_ELAN(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            num_experts=num_experts,
            middle_ratio=0.5,
            num_blocks=2,
        )
        self.norm2 = nn.BatchNorm2d(in_channels)

        # 输出映射层
        # --------------------------------------------------------------
        if out_channels != in_channels:
            self.out_project = Conv(c1=in_channels, c2=out_channels, k=1, act=True)
        else:
            self.out_project = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 注意力子层
        residual = x
        x = self.norm1(x)
        x = self.attn(x) + residual

        # 局部特征提取子层
        residual = x
        x = self.norm2(x)
        x = self.local_extractor(x) + residual

        x = self.out_project(x)

        return x

class TransMoVEV2(nn.Module):
    """采用ELAN结构"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 9,
        kernel_size: int = 3,
        num_blocks: int = 2,
    ):
        super().__init__()
        # 注意力子层
        # --------------------------------------------------------------

        self.attn = DualAxisAggAttn(channels=in_channels)
        self.norm1 = nn.BatchNorm2d(in_channels)
        # 局部特征提取模块
        #  --------------------------------------------------------------
        self.local_extractor = MG_ELAN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_experts=num_experts,
            middle_ratio=0.5,
            num_blocks=num_blocks,
        )
        self.norm2 = nn.BatchNorm2d(in_channels)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 注意力子层
        residual = x
        x = self.norm1(x)
        x = self.attn(x) + residual

        # 局部特征提取子层
        x = self.norm2(x)
        x = self.local_extractor(x)

        return x


if __name__ == "__main__":
    model = TransMoVE(in_channels=64, out_channels=64)
    model_info(model, detailed=False)
    x = torch.randn(1, 64, 224, 224)
    # 将模型设置为评估模式
    model.eval()
    y = model(x)
    print(y.shape)
