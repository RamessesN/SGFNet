import torch
import torch.nn as nn
from einops import rearrange

import torch_dct as dct

class FMFB(nn.Module):
    """
    @brief Frequency Modulated Fusion Block (FMFB).
    
    This module performs adaptive cross-modal interaction in the frequency domain 
    to alleviate spatial misalignment between heterogeneous remote sensing modalities.
    """

    def __init__(self, dim):
        """
        @brief Constructor for the FMFB module.
        
        @param dim The number of input and output channels.
        """
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5

        # 1x1 Convolutions for shared Query and Key generation from frequency difference
        self.conv_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.conv_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # 1x1 Convolutions for independent Value generation
        self.conv_v_h = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.conv_v_x = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # Feed-Forward Networks for feature refinement
        self.ffn_h = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, kernel_size=1)
        )
        self.ffn_x = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, kernel_size=1)
        )

    def forward(self, f_h, f_x):
        """
        @brief Forward pass of the FMFB module.
        
        @param f_h HSI spatial feature tensor of shape (B, C, H, W).
        @param f_x SAR/LiDAR spatial feature tensor of shape (B, C, H, W).
        @return A tuple (out_h, out_x) containing the fusion-modulated spatial features.
        """
        B, C, H, W = f_h.shape

        # 1. Transform spatial features to frequency domain via 2D DCT
        f_h_freq = dct.dct_2d(f_h)
        f_x_freq = dct.dct_2d(f_x)

        # 2. Compute frequency domain difference
        f_d = f_h_freq - f_x_freq

        # 3. Generate shared Query and Key from the difference
        q_d = self.conv_q(f_d)
        k_d = self.conv_k(f_d)

        # 4. Generate independent Values for each modality
        v_h = self.conv_v_h(f_h_freq)
        v_x = self.conv_v_x(f_x_freq)

        # Flatten spatial dimensions for matrix multiplication
        q_d = rearrange(q_d, 'b c h w -> b c (h w)')
        k_d = rearrange(k_d, 'b c h w -> b c (h w)')
        v_h = rearrange(v_h, 'b c h w -> b c (h w)')
        v_x = rearrange(v_x, 'b c h w -> b c (h w)')

        # 5. Compute shared spatial attention map
        attn = (q_d.transpose(-2, -1) @ k_d) * self.scale
        attn = torch.softmax(attn, dim=-1)

        # Apply the shared attention map to both Value representations
        f_h_attn = v_h @ attn
        f_x_attn = v_x @ attn

        # Reshape back to 2D spatial format
        f_h_attn = rearrange(f_h_attn, 'b c (h w) -> b c h w', h=H, w=W)
        f_x_attn = rearrange(f_x_attn, 'b c (h w) -> b c h w', h=H, w=W)

        # 6. Transform modulated features back to spatial domain via 2D IDCT
        f_h_spatial = dct.idct_2d(f_h_attn)
        f_x_spatial = dct.idct_2d(f_x_attn)

        # 7. Refinement via FFN and residual connection
        out_h = self.ffn_h(f_h_spatial) + f_h
        out_x = self.ffn_x(f_x_spatial) + f_x

        return out_h, out_x
