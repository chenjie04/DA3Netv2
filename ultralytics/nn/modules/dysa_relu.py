
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
            channels, channels, 1, groups=1, bias=True
        )

        # --- Spatial Attention (只调制 a) ---
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True), nn.Sigmoid()
        )

        self.sc_mixing = nn.Conv2d(
            channels, channels, 1, groups=1, bias=True
        )

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
        a = torch.sigmoid(self.ch_attn_max(pooled_max))  # [B, C, 1, 1]

        spatial_max = torch.max(x, dim=1, keepdim=True)[0]
        spatial_weight_max = self.spatial_attn(spatial_max)
        a = a * spatial_weight_max

        logit = self.sc_mixing(a)

        out = x * torch.sigmoid(logit)

        return out + x
