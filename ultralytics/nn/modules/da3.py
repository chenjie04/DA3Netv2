import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv


class DualAxisAggAttn_v4(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 4,
        middle_ratio: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.middle_channels = int(in_channels * middle_ratio)

        # 主干 & 短接分支
        self.main_conv = Conv(c1=in_channels, c2=self.middle_channels, k=1, act=True)
        self.short_conv = Conv(c1=in_channels, c2=self.middle_channels, k=1, act=True)

        # QKV 投影（分组）- 确保通道数能被groups整除
        assert self.middle_channels % groups == 0, f"middle_channels ({self.middle_channels}) must be divisible by groups ({groups})"
        self.qkv = nn.ModuleDict({
            "W": nn.Conv2d(
                in_channels=self.middle_channels,
                out_channels=3 * self.middle_channels,
                kernel_size=1,
                groups=groups,
                bias=True,
            ),
            "H": nn.Conv2d(
                in_channels=self.middle_channels,
                out_channels=3 * self.middle_channels,
                kernel_size=1,
                groups=groups,
                bias=True,
            ),
        })

        # 轴向后局部融合（Depthwise）
        self.conv_fusion = nn.ModuleDict({
            "W": Conv(self.middle_channels, self.middle_channels, k=3, g=self.middle_channels),
            "H": Conv(self.middle_channels, self.middle_channels, k=3, g=self.middle_channels),
        })

        # 自适应融合
        self.out_project = Conv(c1=self.middle_channels * 2, c2=out_channels, k=1, act=True)

    def _apply_axis_attention(self, x, axis, external_context=None):
        """轴向上下文聚合 + 门控注入"""
        qkv = self.qkv[axis](x)
        

        query, key, value = torch.split(qkv, [self.middle_channels] * 3, dim=1)

        # 沿轴 softmax
        dim = -1 if axis == "W" else -2
        scores = F.softmax(query, dim=dim)

        # 计算上下文向量
        context = (key * scores).sum(dim=dim, keepdim=True).expand_as(key)


        gate = torch.sigmoid(value)

        # 输出 = 残差 + 门控上下文
        out = x + gate * context.expand_as(x)

        return out, context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        实现全局感受野：
          W轴：每行聚合 → 每个位置获得整行信息
          H轴：每列聚合（输入含行信息）→ 每个位置获得全局信息
        """
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)

        # W轴：沿宽度聚合
        x_W, context_W = self._apply_axis_attention(x_main, "W")
        x_W_fused = self.conv_fusion["W"](x_W) + x_W

        # H轴：沿高度聚合，并接收W轴上下文
        x_H, _ = self._apply_axis_attention(x_W_fused, "H")
        x_H_fused = self.conv_fusion["H"](x_H) + x_H

        # 自适应融合两个分支
        x_fused = torch.cat([x_H_fused, x_short], dim=1)
        x_out = self.out_project(x_fused)

        return x_out

