
import torch
import torch.nn as nn
import torch.nn.functional as F


class DySCA_ReLU(nn.Module):

    def __init__(self, channels: int, reduction: int = 8):
        super(DySCA_ReLU, self).__init__()

        self.channels = channels
        self.expansion = 2  # 输出 (a, b)

        self.extractor = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.norm = nn.BatchNorm2d(channels)

        self.ch_attn_max = nn.Conv2d(
            channels, 1, 1, groups=1, bias=True
        )

        # --- Spatial Attention (只调制 a) ---
        # self.spatial_attn = nn.Sequential(
        #     nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True), nn.Sigmoid()
        # )

        # self.sc_mixing = nn.Conv2d(
        #     channels, channels, 1, groups=1, bias=True
        # )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.extractor(x)
        x = self.norm(x)
        pooled_max = F.max_pool2d(x, (x.size(2), x.size(3)))
        
        spatial_max = torch.max(x, dim=1, keepdim=True)[0]
        # spatial_weight_max = self.spatial_attn(spatial_max)
        a = pooled_max * spatial_max

        logit = self.ch_attn_max(a)

        out = x * torch.sigmoid(logit)

        return out + x

# 有效果
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