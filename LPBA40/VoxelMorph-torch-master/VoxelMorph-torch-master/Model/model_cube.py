'''
VoxelMorph

Original code retrieved from:
https://github.com/voxelmorph/voxelmorph

Original paper:
Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019).
VoxelMorph: a learning framework for deformable medical image registration.
IEEE transactions on medical imaging, 38(8), 1788-1800.

Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
'''

import torch
import torch.nn as nn
import torch.nn.functional as nnf

import numpy as np
from torch.distributions.normal import Normal
import SimpleITK as sitk

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

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

        self.map = nn.Conv3d(self.groups*2, self.groups, 1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.act = nn.GELU()
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(in_channel)

    def forward(self, x):
        H, W, D = self.size
        B, C, H, W, D= x.shape
        shortcut = x
        x_norm = self.norm(x.reshape(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, C, H, W, D)
        # x = x.permute(0, 2, 1)
        x = x_norm.reshape(B * C // self.groups, self.groups, H, W, D)
        x_p = x
        tans_feature = self.TransverseConv(x).view(B * C // self.groups, self.groups, self.size[1], self.size[1],
                                                   self.size[2])
        cor_feature = self.CoronalConv(x).view(B * C // self.groups, self.groups, self.size[1], self.size[0],
                                               self.size[1])

        t1 = tans_feature.permute(0, 1, 3, 2, 4)
        c1 = cor_feature.permute(0, 1, 4, 3, 2)
        f = (c1 @ t1).permute(0, 1, 3, 2, 4)



        shortcut = shortcut.detach().cpu().numpy()
        fusion_feature = (cor_feature @ tans_feature)
        attn = self.attn_drop(fusion_feature.permute(0, 1, 3, 2, 4))
        # if x.shape[2] == 80:
        #     print(x.shape, f.shape, attn.shape)
        #     for i in range(attn.shape[0]):
        #         # f = f.reshape(B * C // self.groups, self.groups, -1).softmax(dim=-1).reshape(B * C // self.groups,
        #         #                                                                              self.groups, H, W, D)
        #         f_cpu = f.detach().cpu().numpy()
        #         attn_cpu = attn.detach().cpu().numpy()
        #
        #         sitk.WriteImage(sitk.GetImageFromArray(f_cpu[i,0, ...]*shortcut[0, 0, ...]), "./%df1.nii"%(i))
        #         sitk.WriteImage(sitk.GetImageFromArray(f_cpu[i, 1, ...]* shortcut[0, 1, ...]), "./%df2.nii"%(i))
        #         sitk.WriteImage(sitk.GetImageFromArray(attn_cpu[i, 0, ...] * shortcut[0, 0, ...]), "./%dattn1.nii"%(i))
        #         sitk.WriteImage(sitk.GetImageFromArray(attn_cpu[i, 1, ...]* shortcut[0, 1, ...]) , "./%dattn2.nii"%(i))
        #
        #         f = f * x
        #     exit()
        f = f.reshape(B * C // self.groups, self.groups, -1).softmax(dim=-1).reshape(B * C // self.groups,
                                                                                     self.groups, H, W, D)
        f = f * x
        x = self.proj_drop(self.act(self.map(torch.cat((attn, f), dim=1))))
        # x = x.reshape(B, C , -1).softmax(dim=-1)
        x = x.reshape(B, C, H, W, D)
        self.flops1 = B * C * self.size[1] * self.size[1] * self.size[2] * self.size[0] * self.size[1]
        self.flops2 = self.flops1
        print((self.flops1 + self.flops2)/1e9)
        # x_cpu = x.reshape(B * C // self.groups, self.groups, -1).softmax(dim=-1).reshape(B * C // self.groups,
        #                                                                              self.groups, H, W, D).detach().cpu().numpy()
        # sitk.WriteImage(sitk.GetImageFromArray(x_cpu[0, 0, ...]+ shortcut[0, 0, ...]) , "./ax1.nii")
        # sitk.WriteImage(sitk.GetImageFromArray(x_cpu[0, 1, ...]+ shortcut[0, 1, ...]) , "./ax2.nii")

        return x

class ConvBlockcube(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, size, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)
        self.size= size
        self.cc = CubeAttention(in_channels, in_channels, size)

    def forward(self, x):
        # print(x.shape, self.size)
        out = self.main(self.cc(x))
        out = self.activation(out)
        return out

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = ((8, 32, 32, 32), (32, 32, 32, 32, 32, 8, 8))

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        idx = 0
        for nf in self.enc_nf:
            self.downarm.append(ConvBlockcube(ndims, prev_nf, nf, size=(inshape[0]//2**idx, inshape[1]//2**idx, inshape[2]//2**idx),stride=2))
            prev_nf = nf
            idx += 1

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            # print(x.shape, x_enc[-1].shape)
            if x.shape[2:] != x_enc[-1].shape[2:]:
                # down 1 dim
                dwc = nn.Conv3d(x.shape[1], x.shape[1], kernel_size=(2,1,1), stride=(1,1,1)).cuda()
                x = dwc(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

class VxmDense_1(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow

class VxmDense_2(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
        inshape,
        nb_unet_features=((8, 24, 24, 24), (24, 24, 24, 16, 16, 8, 8)),
        # nb_unet_features=((16, 32, 32, 32), (32, 32, 32, 32, 32, 16, 16)),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow

class VxmDensex2(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
        inshape,
        nb_unet_features=((32, 64, 64, 64), (64, 64, 64, 64, 64, 32, 32)),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow

class VxmDense_huge(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
        inshape,
        nb_unet_features=((14, 28, 144, 320), (1152, 1152, 320, 144, 28, 14, 14)),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow