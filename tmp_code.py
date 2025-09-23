import torch
import torch.nn as nn
import torch.nn.functional as F

class DySA_ReLU(nn.Module):
    """
    A lightweight version of DySA_ReLU with Gated Residual Connection.
    Simplifications:
    - Lighter gating mechanism (no BatchNorm, higher reduction).
    - Shares the first layer of channel attention networks.
    """

    def __init__(
        self,
        channels: int,
        # Main DySA reduction
        reduction: int = 8, # Increased from default 4 to reduce params
    ):
        super(DySA_ReLU, self).__init__()

        self.channels = channels
        self.expansion = 2 # For (a, b) parameters in each branch

        # --- Shared First Layer for Channel Attention ---
        # Reduces parameters by sharing the initial transformation
        self.shared_ch_attn_down = nn.Conv2d(channels, channels // reduction, 1, bias=True)

        # --- Independent Second Layers for Channel Attention ---
        # Separate transformations after shared pooling for avg and max branches
        mid_channels = channels // reduction
        self.ch_attn_avg_up = nn.Sequential(
            # nn.ReLU(inplace=True), # Could consider removing for more lightweight
            nn.Conv2d(mid_channels, channels * self.expansion, 1, bias=True),
            nn.Sigmoid(), # Initial params in [0, 1]
        )
        self.ch_attn_max_up = nn.Sequential(
            # nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels * self.expansion, 1, bias=True),
            nn.Sigmoid(),
        )

        # --- Shared Spatial Attention Network ---
        # Kept as is for spatial adaptability, relatively lightweight
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(), # Spatial weights in [0, 1]
        )

        # --- Simplified Gate ---
        self.gate = nn.Parameter(torch.tensor(0.0))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Initialize final layers of independent channel attention parts
            if m is self.ch_attn_avg_up[-2]: # Last Conv2d in ch_attn_avg_up
                 nn.init.normal_(m.weight, 0, 0.001)
                 nn.init.constant_(m.bias, 0) # Bias 0 for sigmoid(0)=0.5 -> a=1, b=0
            if m is self.ch_attn_max_up[-2]: # Last Conv2d in ch_attn_max_up
                 nn.init.normal_(m.weight, 0, 0.001)
                 nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Shared First Channel Attention Step ---
        shared_features = self.shared_ch_attn_down(x) # [B, mid_C, H, W]

        # --- Branch 1: Based on Average Pooling ---
        pooled_avg = F.adaptive_avg_pool2d(shared_features, (1, 1)) # [B, mid_C, 1, 1]
        channel_att_avg_params = self.ch_attn_avg_up(pooled_avg) # [B, 2*C, 1, 1]
        a1_init, b1_init = torch.split(channel_att_avg_params, self.channels, dim=1)
        a1_init = a1_init * 2.0 # [0, 1] -> [0, 2]
        b1_init = (b1_init - 0.5) # [0, 1] -> [-0.5, 0.5]

        spatial_avg = torch.mean(x, dim=1, keepdim=True) # [B, 1, H, W]
        spatial_att_avg_params = self.spatial_attn(spatial_avg) # [B, 1, H, W]
        # Apply spatial attention to both a and b
        a1 = a1_init * spatial_att_avg_params # Broadcasting to [B, C, H, W]
        b1 = b1_init * spatial_att_avg_params # Broadcasting to [B, C, H, W]

        # --- Branch 2: Based on Max Pooling ---
        pooled_max = F.adaptive_max_pool2d(shared_features, (1, 1)) # [B, mid_C, 1, 1]
        channel_att_max_params = self.ch_attn_max_up(pooled_max) # [B, 2*C, 1, 1]
        a2_init, b2_init = torch.split(channel_att_max_params, self.channels, dim=1)
        a2_init = a2_init * 2.0 # [0, 1] -> [0, 2]
        b2_init = (b2_init - 0.5) # [0, 1] -> [-0.5, 0.5]

        spatial_max = torch.max(x, dim=1, keepdim=True)[0] # [B, 1, H, W]
        spatial_att_max_params = self.spatial_attn(spatial_max) # Shared network [B, 1, H, W]
        # Apply spatial attention to both a and b
        a2 = a2_init * spatial_att_max_params # Broadcasting to [B, C, H, W]
        b2 = b2_init * spatial_att_max_params # Broadcasting to [B, C, H, W]

        # --- Dynamic ReLU Computation ---
        branch1_out = x * a1 + b1
        branch2_out = x * a2 + b2
        out = torch.max(branch1_out, branch2_out) # Main dynamic output [B, C, H, W]

        # --- Gate Computation ---
        gate_weight = torch.sigmoid(self.gate) 
        # 3. Apply gate: gate_weight * out + (1 - gate_weight) * x
        final_output = gate_weight * out + (1 - gate_weight) * x

        return final_output

# --- Example Usage ---
if __name__ == "__main__":
    B, C, H, W = 2, 64, 32, 32
    channels = C

    input_tensor = torch.randn(B, C, H, W, requires_grad=True)
    # Instantiate the lightweight version
    light_dy_relu = DySA_ReLU(channels=channels, reduction=8)

    print("Lightweight DySA_ReLU with Gated Residual:")
    print(light_dy_relu)

    output_tensor = light_dy_relu(input_tensor)

    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print(f"Output requires grad: {output_tensor.requires_grad}")

    # --- Simple Gradient Check ---
    try:
        loss = output_tensor.sum()
        loss.backward()
        print("\nGradient check passed: Backward pass successful.")
    except Exception as e:
         print(f"\nGradient check failed: {e}")

    # --- Parameter Count Comparison (Optional) ---
    # You can add code here to count parameters of this module vs the previous one
    # to quantify the reduction. This is a simple way:
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Assuming 'dy_relu' is the instance from the previous example
    # print(f"\nOriginal DySA_ReLU parameters: {count_parameters(dy_relu)}")
    # print(f"Lightweight DySA_ReLU parameters: {count_parameters(light_dy_relu)}")



# ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDTN2jhLaj8rl2iPj1s27fIBmGKcr7iOPmUE4vJrMws5 gxuchenjie04@gmail.com







