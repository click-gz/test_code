import torch
import torch.nn as nn
from .IntmdSequential import IntermediateSequential
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
class CubeAttention(nn.Module):
    def __init__(self, in_channel, out_channel, size, kernel=2, stride=2, pad=0, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.groups = 2
        self.size = size
        self.InChannel = in_channel
        self.OutChannel = out_channel
        # print("size: ", size)
        self.TransverseConv = nn.Conv3d(self.groups, (size[1]) * self.groups, (size[0], 3, 3), (1, 1, 1),
                                        padding=(0, 1, 1))
        self.CoronalConv = nn.Conv3d(self.groups, (size[1]) * self.groups, (3, 3, size[2]), (1, 1, 1),
                                     padding=(1, 1, 0))
        self.map = nn.Conv3d(self.groups * 2, self.groups, 3, 1, 1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.act = nn.GELU()
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(in_channel)
        self.pwconv1 = nn.Conv3d(in_channel, 4 * in_channel, kernel_size=1, groups=in_channel)
        self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        self.pwconv2 = nn.Conv3d(4 * in_channel, out_channel, kernel_size=1, groups=in_channel)

        self.drop_path = DropPath(proj_drop) if proj_drop > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(in_channel)

    def forward(self, x):
        H, W, D = self.size
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D)
        shortcut = x
        # x = x.permute(0, 2, 1)
        x = x.reshape(B * C // self.groups, self.groups, H, W, D)
        tans_feature = self.TransverseConv(x).view(B * C // self.groups, self.groups, self.size[1], self.size[1],
                                                   self.size[2])
        cor_feature = self.CoronalConv(x).view(B * C // self.groups, self.groups, self.size[1], self.size[0],
                                               self.size[1])

        t1 = tans_feature.permute(0, 1, 3, 2, 4)
        c1 = cor_feature.permute(0, 1, 4, 3, 2)
        f = (c1 @ t1).permute(0, 1, 3, 2, 4)
        f = f.reshape(B * C // self.groups, self.groups, -1).softmax(dim=-1).reshape(B * C // self.groups,
                                                                                     self.groups, H, W, D)
        f = f * x

        fusion_feature = (cor_feature @ tans_feature)
        attn = self.attn_drop(fusion_feature.permute(0, 1, 3, 2, 4))
        x = self.proj_drop(self.act(self.map(torch.cat((attn, f), dim=1))))
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        return x

class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        size=12
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            # SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                            CubeAttention(
                                dim,
                                dim,
                                to_3tuple(size),
                                attn_drop=attn_dropout_rate,
                                proj_drop=attn_dropout_rate
                            )
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)
        self.depth = depth

    def forward(self, x):
        # print(x.shape, self.depth)
        return self.net(x)
