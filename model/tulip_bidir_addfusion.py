import torch
import torch.nn as nn
import torch.nn.functional as func

from einops import rearrange
from typing import Optional, Tuple

from functools import partial
from util.filter import *

from util.evaluation import inverse_huber_loss
import collections.abc

from mamba_ssm import Mamba

# Only used in Swin, so Not used here
# class DropPath(nn.Module):
#     def __init__(self, drop_prob: float = 0.):
#         super().__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         if self.drop_prob == 0. or not self.training:
#             return x

#         keep_prob = 1 - self.drop_prob
#         shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#         random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#         random_tensor.floor_()
#         x = x.div(keep_prob) * random_tensor
#         return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(16, 1024), 
                 patch_size=(1, 4), 
                 in_c: int = 1, 
                 embed_dim: int = 96, 
                 norm_layer: nn.Module = None, 
                 circular_padding: bool = False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.circular_padding = circular_padding
        if circular_padding:
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(self.patch_size[0], 8), stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)


        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def padding(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            x = func.pad(x, (0, self.patch_size[0] - W % self.patch_size[1],
                             0, self.patch_size[1] - H % self.patch_size[0],
                             0, 0))
        return x

    # Circular padding is only used on the width of range image
    def circularpadding(self, x: torch.Tensor) -> torch.Tensor:
        x = func.pad(x, (2, 2, 0, 0), "circular")
        return x

    def forward(self, x):
        x = self.padding(x)
        #print('After padding:' + str(x.shape))
        if self.circular_padding:
            # Circular Padding
            x = self.circularpadding(x)
            #print('After cir padding:' + str(x.shape))

        x = self.proj(x)
        #print('After proj:' + str(x.shape))
        x = rearrange(x, 'B C H W -> B H W C')
        #print('After rearrange:' + str(x.shape))
        x = self.norm(x)
        #print('After norm:' + str(x.shape))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    @staticmethod
    def padding(x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        if H % 2 == 1 or W % 2 == 1:
            x = func.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        return x


    @staticmethod
    def merging(x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    def forward(self, x):
        x = self.padding(x)
        x = self.merging(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x

# Patch Unmerging layer
class PatchUnmerging(nn.Module):
    def __init__(self, dim: int):
        super(PatchUnmerging, self).__init__()
        self.dim = dim
        #ToDo: Use linear with norm layer?
        self.expand = nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=(1, 1))
        self.upsample = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x: torch.Tensor):
        x = rearrange(x, 'B H W C -> B C H W')
        x = self.expand(x.contiguous())
        x = self.upsample(x)
        # x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=1, P2=4)
        x = rearrange(x, 'B C H W -> B H W C')
        return x

# Original Patch Expanding layer used in Swin MAE
class PatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)
        # self.patch_size = patch_size

    def forward(self, x: torch.Tensor):

        x = self.expand(x)
        # x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=1, P2=4)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=2, P2=2)
        x = self.norm(x)
        return x


# Original Final Patch Expanding layer used in Swin MAE
class FinalPatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm, upscale_factor = 4):
        super(FinalPatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, (4**2) * dim, bias=False)
        self.norm = norm_layer(dim)
        self.upscale_factor = 4

    def forward(self, x: torch.Tensor):
        x = self.expand(x)

        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=self.upscale_factor,
                                                                P2=self.upscale_factor,
                                                                C = self.dim)
        x = self.norm(x)
        return x

class PixelShuffleHead(nn.Module):
    def __init__(self, dim: int, upscale_factor: int):
        super(PixelShuffleHead, self).__init__()
        self.dim = dim

        self.conv_expand = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim*(upscale_factor**2), kernel_size=(1, 1)),
                                         nn.LeakyReLU(inplace=True))


        # self.conv_expand = nn.Conv2d(in_channels=dim, out_channels=dim*(upscale_factor**2), kernel_size=(1, 1))
        self.upsample = nn.PixelShuffle(upscale_factor=upscale_factor)


    def forward(self, x: torch.Tensor):
        x = self.conv_expand(x)
        x = self.upsample(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None,
                 act_layer=nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MlpChannel(nn.Module):
    def __init__(self, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(mlp_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        # 兩個水平掃描方向的卷積核
        self.conv_forward_1234 = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1))  # 順序 1-2-3-4
        self.conv_forward_4321 = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1))  # 反向順序 4-3-2-1

        # Mamba SSM block
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor = dimension of the state vector h_t in Mamba
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.mlp = MlpChannel(dim)

    def forward(self, x):
        token = x
        B, M, N, C = token.shape

        # 將 token reshape 成合適的大小以便進行卷積操作
        token = token.permute(0, 3, 1, 2)  # 轉換為 (B, C, M, N) 以適應 Conv2d

        # forward 1 掃描 (順序 1-2-3-4)
        out_forward_1234 = self.conv_forward_1234(token)  # 水平卷積

        # forward 2 掃描 (反向順序 4-3-2-1)
        out_forward_4321 = self.conv_forward_4321(token.flip(dims=[-1]))  # 水平反向卷積，先翻轉再卷積
        out_forward_4321 = out_forward_4321.flip(dims=[-1])  # 卷積後再次翻轉還原順序

        # 將卷積後的結果還原為原來的形狀
        out_forward_1234 = out_forward_1234.permute(0, 2, 3, 1)  # 轉回 (B, M, N, C)
        out_forward_4321 = out_forward_4321.permute(0, 2, 3, 1)  # 轉回 (B, M, N, C)

        # 將卷積後的輸出 reshape 成 (B, M*N, C) 以便進入後續處理
        token_forward_1234 = out_forward_1234.reshape(B, M * N, C)
        token_forward_4321 = out_forward_4321.reshape(B, M * N, C)

        # 標準化
        token_forward_1234 = self.norm(token_forward_1234)
        token_forward_4321 = self.norm(token_forward_4321)

        # 使用 Mamba SSM 進行後續處理
        out_ssm_1234 = self.mamba(token_forward_1234)
        out_ssm_4321 = self.mamba(token_forward_4321)

        # 將兩個方向的 SSM 輸出結果相加
        out_ssm = out_ssm_1234 + out_ssm_4321 + token_forward_1234 + token_forward_4321
        # BUG FIX: DO NOT overwrite with uni-direction results
        # out_ssm = out_ssm_1234

        # 最後的標準化和 MLP 處理
        out = self.norm(out_ssm)
        out = self.mlp(out) + out_ssm

        # 重塑輸出形狀回到原始大小
        out = out.reshape(B, M, N, C)
        return out

# ADD: New Skip Connection Options
# --- NEW: Skip fusion modules -------------------------------------------------
class CrossAttentionFusion(nn.Module):
    """
    Pixel-wise multi-head cross-attention (no bottleneck).
    Q from x; K/V from {x, skip} at the SAME pixel. No spatial mixing.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"`dim` ({dim}) must be divisible by num_heads ({num_heads}).")
        self.dim = dim

        # Per-pixel LN along channels
        self.q_ln  = nn.LayerNorm(dim)
        self.kv_ln = nn.LayerNorm(dim)   # <-- keep at C, not 2C

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(dim, dim, bias = False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x, skip: [B, H, W, C] (NHWC)
        B, H, W, C = x.shape
        assert C == self.dim

        # LN per source *before* stacking (each sees last-dim = C)
        q_src  = self.q_ln(x)           # [B,H,W,C]
        kx_src = self.kv_ln(x)          # [B,H,W,C]
        ks_src = self.kv_ln(skip)       # [B,H,W,C]

        # Build per-pixel sequences
        q  = q_src.reshape(B * H * W, 1, C)                         # [BHW,1,C]
        kv = torch.stack([kx_src, ks_src], dim=3).reshape(B*H*W, 2, C)  # [BHW,2,C]

        # Pixel-wise multi-head attention
        y, _ = self.attn(q, kv, kv, need_weights=False)             # [BHW,1,C]
        y = y.reshape(B, H, W, C)

        # Residual in pixel space
        return self.out_proj(y) + x
    
class CrossMambaFusion(nn.Module):
    """
    Cross-Mamba fusion: interleave tokens (x0, s0, x1, s1, ...)
    run Mamba over length 2L, then take even positions back as fused x. Residual to x.
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.ln_x = nn.LayerNorm(dim)
        self.ln_s = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x, skip: [B, H, W, C]
        B, H, W, C = x.shape
        L = H * W
        x_flat = self.ln_x(x).reshape(B, L, C)
        s_flat = self.ln_s(skip).reshape(B, L, C)

        # interleave along sequence dimension: [B, 2L, C]
        inter = torch.stack((x_flat, s_flat), dim=2).reshape(B, 2 * L, C)

        y = self.mamba(inter)                        # [B, 2L, C]
        y_even = y[:, 0::2, :]                       # take positions of x
        y_even = self.out_proj(y_even) + x_flat      # residual to x

        return y_even.reshape(B, H, W, C)


class SkipFusion(nn.Module):
    """
    - 'linear'      : concat on channels then Linear(dim*2 -> dim) — pixel-wise
    - 'cross_attn'  : pixel-wise multi-head CrossAttentionFusion (no bottleneck)
    - 'cross_mamba' : pixel-wise CrossMambaFusion (no bottleneck)
    """
    def __init__(self, dim: int, mode: str = "linear",
                 attn_heads: int = 4, attn_dropout: float = 0.0,
                 mamba_state: int = 16, mamba_conv: int = 4, mamba_expand: int = 2,
                 ):
        super().__init__()
        mode = mode.lower()
        self.mode = mode
        if mode == "linear":
            self.fuser = nn.Linear(dim * 2, dim)
        elif mode == "cross_attn":
            self.fuser = CrossAttentionFusion(dim, num_heads=attn_heads, dropout=attn_dropout)
        elif mode == "cross_mamba":
            self.fuser = CrossMambaFusion(dim, d_state=mamba_state, d_conv=mamba_conv,
                                          expand=mamba_expand)
        else:
            raise ValueError(f"Unknown skip fusion mode: {mode}")

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear":
            # Pixel-wise linear fusion on channels
            return self.fuser(torch.cat([x, skip], dim=-1))
        else:
            # Pixel-wise cross_* modes consume both tensors and return fused feature (same shape as x)
            return self.fuser(x, skip)
# --------------------------------------------------------------------------

# class MambaLayer(nn.Module):
#     def __init__(self, dim, d_state=16, d_conv=4, expand=2):
#         super().__init__()
#         self.dim = dim
#         self.norm = nn.LayerNorm(dim)
#         self.mamba = Mamba(
#             d_model=dim,  # Model dimension d_model
#             d_state=d_state,  # SSM state expansion factor
#             d_conv=d_conv,  # Local convolution width
#             expand=expand,  # Block expansion factor
#         )
#         self.mlp = MlpChannel(dim)
#
#     def forward(self, x):
#
#             token = x
#
#             B, M, N, C = token.shape
#
#             token = x.reshape(B, M*N, C)
#
#             token = self.norm(token)
#             out_ssm = self.mamba(token) + token
#             out = self.norm(out_ssm)
#             out = self.mlp(out) + out_ssm
#
#             out = out.reshape(B, M, N, C)
#
#             return out


class BasicBlock(nn.Module):
    def __init__(self, index: int,
                embed_dim: int = 96, 
                window_size: int = 7, 
                depths: tuple = (2, 2, 2, 2),
                 num_heads: tuple = (3, 6, 12, 24), 
                 mlp_ratio: float = 4., 
                 qkv_bias: bool = True,
                 drop_rate: float = 0., 
                 attn_drop_rate: float = 0., 
                 drop_path: float = 0.1,
                 norm_layer=nn.LayerNorm, 
                 patch_merging: bool = True):
        super(BasicBlock, self).__init__()
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]

        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]

        self.blocks = nn.ModuleList([
            MambaLayer(
                dim=dim
            )
            for i in range(depth)])

        if patch_merging:
            self.downsample = PatchMerging(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicBlockUp(nn.Module):
    def __init__(self, index: int, embed_dim: int = 96, window_size: int = 7, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 patch_expanding: bool = True, norm_layer=nn.LayerNorm, patch_unmerging: bool = False):
        super(BasicBlockUp, self).__init__()
        index = len(depths) - index - 2
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]

        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]

        self.blocks = nn.ModuleList([
            MambaLayer(
                dim=dim
            )
            for i in range(depth)])
        if patch_expanding:
            if patch_unmerging:
                self.upsample = PatchUnmerging(dim = embed_dim * 2 ** index)
            else:
                self.upsample = PatchExpanding(dim=embed_dim * 2 ** index, norm_layer=norm_layer)

        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.upsample(x)
        return x

class TULIP(nn.Module):
    def __init__(self, img_size = (16, 1024), 
                 target_img_size = (64, 1024) ,
                 patch_size = (1, 4), 
                 in_chans: int = 1, 
                 embed_dim: int = 96,
                 window_size: int = (2, 8), # No effect
                 depths: tuple = (2, 2, 2, 2), # No effect 
                 num_heads: tuple = (3, 6, 12, 24), # No effect
                 mlp_ratio: float = 4., # No effect
                 qkv_bias: bool = True, # No effect
                 drop_rate: float = 0., 
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, # No effect
                 norm_layer=nn.LayerNorm, 
                 patch_norm: bool = True, 
                 pixel_shuffle: bool = True, 
                 circular_padding: bool = True, 
                 swin_v2: bool = False, 
                 log_transform: bool = True,
                 patch_unmerging: bool = True,
                 # ADD: more skip connections options
                 # --- NEW: skip fusion configuration ---
                 skip_mode: str = "linear",        # "linear" | "cross_attn" | "cross_mamba"
                 skip_attn_heads: int = 4,
                 skip_attn_dropout: float = 0.0,
                 skip_mamba_state: int = 16,
                 skip_mamba_conv: int = 4,
                 skip_mamba_expand: int = 2
                 ):
        super().__init__()

        self.window_size = window_size
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path = drop_path_rate
        self.norm_layer = norm_layer
        self.img_size = img_size
        self.target_img_size = target_img_size
        self.log_transform = log_transform

        # NEW: stash skip fusion configs
        self.skip_mode = skip_mode
        self.skip_attn_heads = skip_attn_heads
        self.skip_attn_dropout = skip_attn_dropout
        self.skip_mamba_state = skip_mamba_state
        self.skip_mamba_conv = skip_mamba_conv
        self.skip_mamba_expand = skip_mamba_expand

        self.pos_drop = nn.Dropout(p=drop_rate) # 0
        self.patch_unmerging = patch_unmerging
        if swin_v2: # Not used
            self.layers = self.build_layers_v2()
            self.layers_up = self.build_layers_up_v2()
        else:
            self.layers = self.build_layers()
            self.layers_up = self.build_layers_up()

        if self.patch_unmerging:
            self.first_patch_expanding = PatchUnmerging(dim=embed_dim * 2 ** (len(depths) - 1))
        else:
            self.first_patch_expanding = PatchExpanding(dim=embed_dim * 2 ** (len(depths) - 1), norm_layer=norm_layer)


        # NEW EFFECT: this will build differenct skip connection based on self.skip_mode
        self.skip_connection_layers = self.skip_connection()

        self.norm_up = norm_layer(embed_dim)

        self.patch_embed = PatchEmbedding(img_size = img_size, patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                            norm_layer=norm_layer if patch_norm else None, circular_padding=circular_padding)

        self.decoder_pred = nn.Conv2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(1, 1), bias=False)

        self.pixel_shuffle = pixel_shuffle
        self.upscale_factor = int(((target_img_size[0]*target_img_size[1]) / (img_size[0]*img_size[1]))**0.5) * 2 * int(((patch_size[0]*patch_size[1])//4)**0.5)

        if self.pixel_shuffle:
            self.ps_head = PixelShuffleHead(dim = embed_dim, upscale_factor=self.upscale_factor)
        else:
            self.final_patch_expanding = FinalPatchExpanding(dim=embed_dim, norm_layer=norm_layer, upscale_factor=self.upscale_factor)

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths, # (2, 2, 2, 2)
                embed_dim=self.embed_dim,# 96
                num_heads=self.num_heads, # (3, 6, 12, 24)
                drop_path=self.drop_path,  # 0.1
                window_size=self.window_size, # No effect
                mlp_ratio=self.mlp_ratio, # 4
                qkv_bias=self.qkv_bias, # No effect
                drop_rate=self.drop_rate, # 0
                attn_drop_rate=self.attn_drop_rate, # 0
                norm_layer=self.norm_layer, # nn.LayerNorm
                patch_merging=False if i == self.num_layers - 1 else True,)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                patch_unmerging=self.patch_unmerging)
            layers_up.append(layer)
        return layers_up

    #  --------------- MODIFY: add more skip connections options
    def skip_connection(self):
        """Build per-stage skip fusion modules according to configured mode."""
        skip_connection_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            dim = self.embed_dim * 2 ** (self.num_layers - 2 - i)
            if self.skip_mode == "linear":
                layer = SkipFusion(dim, mode="linear")
            elif self.skip_mode == "cross_attn":
                layer = SkipFusion(dim, mode="cross_attn",
                                   attn_heads=self.skip_attn_heads,
                                   attn_dropout=self.skip_attn_dropout)
            elif self.skip_mode == "cross_mamba":
                layer = SkipFusion(dim, mode="cross_mamba",
                                   mamba_state=self.skip_mamba_state,
                                   mamba_conv=self.skip_mamba_conv,
                                   mamba_expand=self.skip_mamba_expand)
            else:
                raise ValueError(f"Unknown skip_mode: {self.skip_mode}")
            skip_connection_layers.append(layer)
        return skip_connection_layers

    def forward_loss(self, pred, target):

        loss = (pred - target).abs()
        loss = loss.mean()

        if self.log_transform:
            pixel_loss = (torch.expm1(pred) - torch.expm1(target)).abs().mean()
        else:
            pixel_loss = loss.clone()

        return loss, pixel_loss

    def forward(self, x, target, eval = False, mc_drop = False):

        x = self.patch_embed(x)
        x = self.pos_drop(x) # 0
        x_save = []
        for i, layer in enumerate(self.layers):
            x_save.append(x)
            x = layer(x)

        x = self.first_patch_expanding(x) # Patch Unmerging


        for i, layer in enumerate(self.layers_up):
            # MODIFY: more skip connection options
            skip_feat = x_save[len(x_save) - i - 2]           # encoder feature to skip
            # --- NEW: use configured skip fusion instead of fixed concat+Linear ---
            x = self.skip_connection_layers[i](x, skip_feat)  # returns [B, H, W, dim]
            x = layer(x)


        x = self.norm_up(x)


        if self.pixel_shuffle:
            x = rearrange(x, 'B H W C -> B C H W')
            x = self.ps_head(x.contiguous())
        else:
            x = self.final_patch_expanding(x)
            x = rearrange(x, 'B H W C -> B C H W')


        x = self.decoder_pred(x.contiguous())

        if mc_drop:
            return x
        else:
            total_loss, pixel_loss = self.forward_loss(x, target)
            return x, total_loss, pixel_loss

#MODIFY: change name for easire import
def tulip_base_mamba(**kwargs):
    model = TULIP(
        depths=(2, 2, 2, 2), 
        embed_dim=96, 
        num_heads=(3, 6, 12, 24),
        qkv_bias=True,
        mlp_ratio=4,
        drop_path_rate=0.1, 
        drop_rate=0, 
        attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        skip_mode = "linear", 
        **kwargs)
        
    return model

def tulip_base_mamba_crsatten(**kwargs):
    model = TULIP(
        depths=(2, 2, 2, 2), 
        embed_dim=96, 
        num_heads=(3, 6, 12, 24),
        qkv_bias=True,
        mlp_ratio=4,
        drop_path_rate=0.1, 
        drop_rate=0, 
        attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        skip_mode = "cross_attn", 
        skip_attn_heads = 4,
        skip_attn_dropout = 0.0,
        **kwargs)
        
    return model

def tulip_base_mamba_crsmamba(**kwargs):
    model = TULIP(
        depths=(2, 2, 2, 2), 
        embed_dim=96, 
        num_heads=(3, 6, 12, 24),
        qkv_bias=True,
        mlp_ratio=4,
        drop_path_rate=0.1, 
        drop_rate=0, 
        attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        skip_mode = "cross_mamba", 
        skip_mamba_state = 16,
        skip_mamba_conv = 4,
        skip_mamba_expand = 2,
        **kwargs)
        
    return model






