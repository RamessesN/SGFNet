import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

class SMCB(nn.Module):
    """
    @brief Semantic Mixing Convolution Block (SMCB).
    
    This module dynamically generates semantic-aware convolution kernels based on 
    contextual affinities among feature representations, enhancing semantic dependency 
    modeling and adaptive feature aggregation.
    """

    def __init__(self, dim, kernel_size=3):
        """
        @brief Constructor for the SMCB module.
        
        @param dim The number of input and output channels.
        @param kernel_size The size of the dynamic convolution kernel. Default is 3.
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.k_sq = kernel_size ** 2

        # Value branch: 1x1 Convolution
        self.conv_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # Query branch: 1x1 Convolution
        self.conv_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # Key branch: Adaptive Average Pooling followed by 1x1 Convolution
        self.adapool = nn.AdaptiveAvgPool2d((kernel_size, kernel_size))
        self.conv_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # Dynamic kernel generation: Linear mapping
        self.kernel_proj = nn.Linear(self.k_sq, self.k_sq)

        # Local feature enhancement branch: 3x3 Depth-wise Convolution
        self.dconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

    def forward(self, x):
        """
        @brief Forward pass of the SMCB module.
        
        @param x Input tensor of shape (B, C, H, W).
        @return Output tensor of shape (B, C, H, W) after semantic-guided feature fusion.
        """
        B, C, H, W = x.shape

        # 1. Extract Query features
        q = self.conv_q(x)
        q = rearrange(q, 'b c h w -> b c (h w)')

        # 2. Extract Key features
        k = self.adapool(x)
        k = self.conv_k(k)
        k = rearrange(k, 'b c h w -> b c (h w)')

        # 3. Compute Affinity map
        affinity = einsum(q, k, 'b c n, b c l -> b n l')

        # 4. Generate Dynamic kernel
        dynamic_kernel = self.kernel_proj(affinity)
        dynamic_kernel = torch.softmax(dynamic_kernel, dim=-1)
        dynamic_kernel = rearrange(dynamic_kernel, 'b (h w) k -> b 1 h w k', h=H, w=W)

        # 5. Extract Value features and unfold into local patches
        v = self.conv_v(x)
        v_unfold = F.unfold(v, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        v_unfold = rearrange(v_unfold, 'b (c k) (h w) -> b c h w k', c=C, k=self.k_sq, h=H, w=W)

        # 6. Aggregate features via dynamic convolution
        x_prime = einsum(dynamic_kernel, v_unfold, 'b 1 h w k, b c h w k -> b c h w')

        # 7. Extract local enhanced features via parallel depth-wise convolution
        x_d = self.dconv(x)

        # 8. Perform residual fusion
        out = x_prime + x_d

        return out
