# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from monai.utils import optional_import

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")


class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.flops1 = 0
        self.flops2 = 0

    def forward(self, x):
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        self.flops1 = q.shape[0]*q.shape[1]*q.shape[2]*q.shape[3]*k.shape[2]
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        self.flops2 = att_mat.shape[0]*att_mat.shape[1]*att_mat.shape[2]*att_mat.shape[3]*v.shape[2]
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x

    def flops(self):
        # calculate flops for 1 window with token length of N
        flops = 0

        # attn = (q @ k.transpose(-2, -1))
        flops += self.flops1
        #  x = (attn @ v)
        flops += self.flops2

        return flops

class CubeAttention(nn.Module):
    def __init__(self, in_channel, out_channel, size, kernel=2, stride=2, pad=0, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.groups = 2
        self.size = size
        self.InChannel = in_channel
        self.OutChannel = out_channel
        # print("size: ", size)
        self.TransverseConv = nn.Conv3d(self.groups, (size[1])*self.groups, (size[0], 3, 3), (1, 1, 1),
                                        padding=(0, 1, 1))
        self.CoronalConv = nn.Conv3d(self.groups, (size[1])*self.groups, (3, 3, size[2]), (1, 1, 1),
                                     padding=(1, 1, 0))
        self.map = nn.Conv3d(in_channel*2, in_channel, 1)
        self.mpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.act = nn.GELU()
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(in_channel)
        self.flops1 = 0
        self.flops2 = 0
    def forward(self, x):
        H, W, D = self.size
        B, N, C= x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D)
        shortcut = x
        # x = x.permute(0, 2, 1)
        x = x.reshape(B * C//self.groups, self.groups, H, W, D)
        tans_feature = self.TransverseConv(x).view(B, C, self.size[1], self.size[1], self.size[2])
        cor_feature = self.CoronalConv(x).view(B, C, self.size[1], self.size[0], self.size[1])
        self.flops1 = B*C*self.size[1]*self.size[1]*self.size[2]*self.size[0]*self.size[1]
        self.flops2 = self.flops1
        t1 = tans_feature.permute(0, 1, 3, 2, 4)
        c1 = cor_feature.permute(0, 1, 4, 3, 2)
        f = (c1@t1).permute(0, 1, 3, 2, 4)


        fusion_feature = (cor_feature @ tans_feature)
        attn = self.attn_drop(fusion_feature.permute(0, 1, 3, 2, 4))

        x = self.proj_drop(self.act(self.map(torch.cat((shortcut*f, attn), dim=1))))
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        return x
    def flops(self):
        return self.flops2+self.flops1


class CubeAttention1(nn.Module):
    def __init__(self, in_channel, out_channel, size, kernel=2, stride=2, pad=0, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.groups = 2
        self.size = size
        self.InChannel = in_channel
        self.OutChannel = out_channel
        # print("size: ", size)
        self.TransverseConv = nn.Conv3d(self.groups, (size[1])*self.groups, (size[0], 3, 3), (1, 1, 1),
                                        padding=(0, 1, 1))
        self.CoronalConv = nn.Conv3d(self.groups, (size[1])*self.groups, (3, 3, size[2]), (1, 1, 1),
                                     padding=(1, 1, 0))
        self.map = nn.Conv3d(self.groups*2, self.groups, 3, 1, 1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.act = nn.GELU()
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(in_channel)

    def forward(self, x):
        H, W, D = self.size
        B, N, C= x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D)
        shortcut = x
        # x = x.permute(0, 2, 1)
        x = x.reshape(B * C//self.groups, self.groups, H, W, D)
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
