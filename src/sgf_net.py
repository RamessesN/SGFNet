import torch
import torch.nn as nn

from modules.smcb_block import SMCB
from modules.fmfb_block import FMFB

class SGFM(nn.Module):
    """
    @brief Semantic-Guided Fusion Module (SGFM) with block-level residual connections.
    
    This module contains two SMCB blocks and one FMFB block. It integrates global 
    block-level residual shortcuts to prevent gradient degradation during deep fusion.
    """
    def __init__(self, dim):
        """
        @brief Constructor for the SGFM module.
        
        @param dim The number of input and output channels for both modalities.
        """
        super().__init__()
        self.smcb_h = SMCB(dim)
        self.smcb_x = SMCB(dim)
        self.fmfb = FMFB(dim)
        
        # Trailing refinement layers for both branches
        self.conv_h = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn_h = nn.BatchNorm2d(dim)
        self.conv_x = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn_x = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, f_h, f_x):
        """
        @brief Forward pass of the SGFM module with block-level residuals.
        
        @param f_h HSI feature tensor of shape (B, dim, H, W).
        @param f_x SAR/LiDAR feature tensor of shape (B, dim, H, W).
        @return A tuple (out_h, out_x) containing enhanced features with residual addition.
        """
        # Parallel semantic context modeling
        h_smcb = self.smcb_h(f_h)
        x_smcb = self.smcb_x(f_x)
        
        # Frequency domain cross-modal interaction
        h_fmfb, x_fmfb = self.fmfb(f_h, f_x)
        
        # Intra-block feature aggregation
        h_fused = h_smcb + h_fmfb
        x_fused = x_smcb + x_fmfb
        
        # Convolutional refinement
        h_out = self.act(self.bn_h(self.conv_h(h_fused)))
        x_out = self.act(self.bn_x(self.conv_x(x_fused)))
        
        # Block-level global residual connections
        return h_out + f_h, x_out + f_x

class SGFNet(nn.Module):
    """
    @brief Semantic-Guided Fusion Network (SGFNet) with Bottleneck fusion strategy.
    
    A hierarchical model that processes HSI and SAR/LiDAR tokens through stacked 
    SGFMs, fuses them via channel concatenation and a bottleneck layer, and outputs 
    classification maps.
    """
    def __init__(self, hsi_bands=30, sar_channels=1, num_classes=7, dim=64, num_sgfm=2):
        """
        @brief Constructor for the SGFNet model.
        
        @param hsi_bands Number of principal components for HSI input. Default is 30.
        @param sar_channels Number of input channels for SAR/LiDAR. Default is 1.
        @param num_classes Number of target land-cover categories. Default is 7.
        @param dim Internal feature dimension size. Default is 64.
        @param num_sgfm Number of stacked SGFM blocks. Default is 2.
        """
        super().__init__()
        
        # HSI initial 3D spectral-spatial extraction
        self.hsi_init = nn.Sequential(
            nn.Conv3d(1, dim, kernel_size=(hsi_bands, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(dim),
            nn.GELU()
        )
        
        # SAR/LiDAR initial 2D structural extraction
        self.sar_init = nn.Sequential(
            nn.Conv2d(sar_channels, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # Stacked fusion blocks
        self.sgfm_blocks = nn.ModuleList([
            SGFM(dim) for _ in range(num_sgfm)
        ])
        
        # Cross-modal channel mixing bottleneck
        self.fusion_bottleneck = BottleneckLayer(dim * 2, dim)
        
        # Final classification refinement layers
        self.refinement = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, num_classes, kernel_size=1)
        )

    def forward(self, x_hsi, x_sar):
        """
        @brief Forward pass of the complete SGFNet architecture.
        
        @param x_hsi Input HSI tensor of shape (B, 1, hsi_bands, H, W).
        @param x_sar Input SAR/LiDAR tensor of shape (B, sar_channels, H, W).
        @return Classification logit tensor of shape (B, num_classes, H, W).
        """
        # HSI initial projection and dimension squeezing
        f_h = self.hsi_init(x_hsi)
        f_h = f_h.squeeze(2) 
        
        # SAR/LiDAR initial projection
        f_x = self.sar_init(x_sar)
        
        # Hierarchical SGFM feature processing
        for block in self.sgfm_blocks:
            f_h, f_x = block(f_h, f_x)
            
        # Channel-wise concatenation followed by bottleneck fusion
        f_fused = torch.cat([f_h, f_x], dim=1)
        f_fused = self.fusion_bottleneck(f_fused)
        
        # Generate classification map
        out = self.refinement(f_fused)
        return out
    
class BottleneckLayer(nn.Module):
    """
    @brief Standard Bottleneck Layer for feature fusion.
    
    This module implements a 1x1 -> 3x3 -> 1x1 convolution sequence with a residual 
    shortcut to mix concatenated channel features efficiently and reduce dimensions.
    """
    def __init__(self, in_channels, out_channels):
        """
        @brief Constructor for the BottleneckLayer module.
        
        @param in_channels The number of channels of the concatenated input tensor.
        @param out_channels The number of output channels after fusion.
        """
        super().__init__()
        hidden_dim = out_channels // 2
        
        # 1x1 Reduction layer
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # 3x3 Spatial feature extraction layer
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # 1x1 Expansion layer
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.act = nn.GELU()
        
        # Residual shortcut to match channel dimensions
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        """
        @brief Forward pass of the BottleneckLayer module.
        
        @param x Concatenated input tensor of shape (B, in_channels, H, W).
        @return Fused and dimension-reduced tensor of shape (B, out_channels, H, W).
        """
        identity = self.shortcut(x)
        
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out = out + identity
        return self.act(out)
