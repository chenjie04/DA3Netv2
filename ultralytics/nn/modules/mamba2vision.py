

from ast import main
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torch.nn.qat import Conv2d

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from ultralytics.nn.modules.conv import Conv

# 0.54
def spatial_channels_attention(x: torch.Tensor) -> torch.Tensor:
    # 聚合全局特征
    context = F.adaptive_avg_pool2d(x, 1)
    # 计算每个位置与全局特征之间的相似度
    logits = x * context
    # 归一化
    weights = F.sigmoid(logits)
    # 乘以权重
    out = x * weights
    return out


# 0.549
# class DYReLUC(nn.Module):
#     """
#     DY-ReLU-C: Dynamic ReLU with Spatial and Channel-wise Attention

#     Args:
#         channels (int): 输入通道数
#         reduction (int): 通道注意力的降维比例，默认为4
#     """

#     def __init__(
#         self,
#         channels: int,
#         reduction: int = 4,
#     ):
#         super(DYReLUC, self).__init__()

#         self.channels = channels
#         self.expansion = 4  # for a1, b1, a2, b2

#         # 通道注意力模块
#         mid_channels = max(channels // reduction, 1)
#         self.channel_attn = nn.Sequential(
#             nn.Conv2d(channels, mid_channels, 1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, channels * self.expansion, 1, bias=True),
#             nn.Hardsigmoid(),
#         )

#         # 空间注意力模块
#         self.spatial_attn = nn.Sequential(
#             nn.Conv2d(
#                 channels, 1 * 2, kernel_size=1, padding=0, bias=True
#             ),
#             nn.Hardsigmoid(),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         前向传播

#         Args:
#             x (torch.Tensor): 输入张量，形状为 [B, C, H, W]

#         Returns:
#             torch.Tensor: 输出张量，形状为 [B, C, H, W]
#         """

#         # 全局平均池化获取通道注意力权重
#         squeeze = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]

#         # 通道注意力
#         coeffs = (
#             self.channel_attn(squeeze) - 0.5
#         )  # [B, C*4, 1, 1] value range [-0.5, 0.5]
#         a1, b1, a2, b2 = torch.split(coeffs, self.channels, dim=1)
#         a1 = a1 * 2.0 + 1.0
#         a2 = a2 * 2.0

#         # 空间注意力
#         attn = self.spatial_attn(x)  # [B, 4, H, W]
#         a1_s, a2_s = torch.split(attn, 1, dim=1)

#         a1, a2 = a1 * a1_s, a2 * a2_s

#         out = torch.max(x * a1 + b1, x * a2 + b2)

#         return out

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
    ):
        super().__init__()
        self.channels = channels

        self.qkv = nn.ModuleDict(
            {
                "W": nn.Conv2d(
                    in_channels=channels,
                    out_channels=1 + 2 * channels,
                    kernel_size=1,
                    bias=True,
                ),
                "H": nn.Conv2d(
                    in_channels=channels,
                    out_channels=1 + 2 * channels,
                    kernel_size=1,
                    bias=True,
                ),
            }
        )

        self.conv_fusion = nn.ModuleDict(
            {
                "W": nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True),
                "H": nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, bias=True),
            }
        )

    def _apply_axis_attention(self, x, axis):
        """通用轴注意力计算"""
        qkv = self.qkv[axis](x)
        query, key, value = torch.split(
            qkv, [1, self.channels, self.channels], dim=1
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
        out = self.conv_fusion[axis](out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, H, W]

        Returns:
            torch.Tensor: 输出张量，形状为 [B, C, H, W]
        """
        # W轴注意力
        x = self._apply_axis_attention(x, "W")
        # H轴注意力
        x = self._apply_axis_attention(x, "H")
       
        return x

class Mamba2Vision(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=1,
        headdim=128,
        ngroups=2,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.middle_dim = d_model // 2
        self.d_state = d_state
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.middle_dim
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        self.short_conv = light_ConvBlock(d_model, self.d_inner)
        
        self.main_conv = Conv(d_model, self.d_inner,k=1, act=True)
        self.dual_axis_attn = DualAxisAggAttn(channels=self.d_inner)

        # Order: [x, B, C, dt]
        d_in_proj = self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_inner, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Conv2d(self.d_inner * 2, self.d_model, kernel_size=1)

    def forward(self, u, seq_idx=None):
        """
        u: (B, C, H, W)
        Returns: same shape as u
        """

        B, C, H, W = u.shape

        z = self.short_conv(u)

        u = self.main_conv(u)
        u = self.dual_axis_attn(u)
        u = rearrange(u, "b c h w -> b (h w) c").contiguous()
        batch, seqlen, dim = u.shape

        xbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        xBC, dt = torch.split(
            xbcdt, [self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        assert self.activation in ["silu", "swish"]

        # 1D Convolution
        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
        )  # (B, L, self.d_inner + 2 * ngroups * d_state)
        xBC = xBC[:, :seqlen, :]


        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y)

        # Reshape to 2D dimensions (b, l, d) -> (b, d, h, w)
        y = rearrange(y, "b (h w) d -> b d h w", h=H, w=W).contiguous()
        y = torch.cat([z, y], dim=1)
        out = self.out_proj(y)
        return out

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=["avg", "max"]):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        channel_att_raw = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(
                    x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == "lse":
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = Conv(c1=2, c2=1, k=kernel_size, act=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


class DyGLU(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 4,
    ):
        super().__init__()
        self.channels = channels

        self.channel_gate = ChannelGate(
            gate_channels=channels, reduction_ratio=reduction, pool_types=["avg", "max"]
        )
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)

        return x


if __name__ == "__main__":
   x = torch.rand(6, 128, 80, 80).cuda()
   model = Mamba2Vision(d_model=128, headdim=32).cuda()
   out = model(x)
   print(out.shape)