# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, ContextBlock
from torch.nn import functional as F
from mmengine.model import constant_init
import math
from ..builder import HEADS
from ..utils import resize
from .decode_head import BaseDecodeHead

class _SelfAttentionBlock(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out,
                 conv_cfg, norm_cfg, act_cfg):
        super(_SelfAttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.value_project = self.build_project(
            key_in_channels,
            channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context



class SelfAttentionBlock(_SelfAttentionBlock):
    def __init__(self, in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        super(SelfAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=True,
            with_out=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.output_project = self.build_project(
            in_channels,
            in_channels,
            num_convs=1,
            use_conv_module=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        context = super(SelfAttentionBlock, self).forward(x, x)
        return self.output_project(context)


class ISA_atten(nn.Module):
    def __init__(self, isa_channels, in_channels, channels, down_factor=(8, 8)):
        super(ISA_atten, self).__init__()
        self.down_factor = down_factor
        self.in_channels = in_channels
        self.channels = channels
        self.norm_cfg = dict(type='BN', requires_grad=True)
        self.act_cfg = dict(type='ReLU')
        self.conv_cfg = None

        self.in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.global_relation = SelfAttentionBlock(
            self.channels,
            isa_channels,
            conv_cfg = None,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.local_relation = SelfAttentionBlock(
            self.channels,
            isa_channels,
            norm_cfg=self.norm_cfg,
            conv_cfg=self.conv_cfg,
            act_cfg=self.act_cfg)
        self.out_conv = ConvModule(
            self.channels * 2,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)


    def forward(self, inputs):
        """Forward function."""
        x_ = inputs
        x = self.in_conv(x_)
        residual = x
        n, c, h, w = x.size()
        loc_h, loc_w = self.down_factor  # size of local group in H- and W-axes
        glb_h, glb_w = math.ceil(h / loc_h), math.ceil(w / loc_w)
        pad_h, pad_w = glb_h * loc_h - h, glb_w * loc_w - w
        if pad_h > 0 or pad_w > 0:  # pad if the size is not divisible
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                       pad_h - pad_h // 2)
            x = F.pad(x, padding)

        # global relation
        x = x.view(n, c, glb_h, loc_h, glb_w, loc_w)
        # do permutation to gather global group
        x = x.permute(0, 3, 5, 1, 2, 4)  # (n, loc_h, loc_w, c, glb_h, glb_w)
        x = x.reshape(-1, c, glb_h, glb_w)
        # apply attention within each global group
        x = self.global_relation(x)  # (n * loc_h * loc_w, c, glb_h, glb_w)

        # local relation
        x = x.view(n, loc_h, loc_w, c, glb_h, glb_w)
        # do permutation to gather local group
        x = x.permute(0, 4, 5, 3, 1, 2)  # (n, glb_h, glb_w, c, loc_h, loc_w)
        x = x.reshape(-1, c, loc_h, loc_w)
        # apply attention within each local group
        x = self.local_relation(x)  # (n * glb_h * glb_w, c, loc_h, loc_w)

        # permute each pixel back to its original position
        x = x.view(n, glb_h, glb_w, c, loc_h, loc_w)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (n, c, glb_h, loc_h, glb_w, loc_w)
        x = x.reshape(n, c, glb_h * loc_h, glb_w * loc_w)
        if pad_h > 0 or pad_w > 0:  # remove padding
            x = x[:, :, pad_h // 2:pad_h // 2 + h, pad_w // 2:pad_w // 2 + w]
        x = self.out_conv(torch.cat([x, residual], dim=1))
        return x



class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners, **kwargs):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        **kwargs)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


@HEADS.register_module()
class OrgSeg_Head(BaseDecodeHead):
    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(OrgSeg_Head, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)

        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)   

        self.fusion = ConvModule(
            self.channels + self.in_channels[-3]//4 +  self.in_channels[-4]//4 + self.in_channels[-2]//4,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # 1024-> 1024 // 4
        self.context_3 = nn.Sequential(
            DepthwiseSeparableConvModule(in_channels=self.in_channels[-2],
                                        out_channels=self.in_channels[-2]//4,
                                        kernel_size=3,
                                        padding=1), 

            DepthwiseSeparableConvModule(in_channels=self.in_channels[-2]//4,
                                        out_channels=self.in_channels[-2],
                                        kernel_size=3,
                                        padding=1),    

            DepthwiseSeparableConvModule(in_channels=self.in_channels[-2],
                                        out_channels=self.in_channels[-2]//4,
                                        kernel_size=3,
                                        padding=1),

            ContextBlock(in_channels = self.in_channels[-2]//4, ratio=0.4)
        )          
        # 512-> 512
        self.context_2 = nn.Sequential(
            DepthwiseSeparableConvModule(in_channels=self.in_channels[-3],
                                        out_channels=self.in_channels[-3]//4,
                                        kernel_size=3,
                                        padding=1), 

            DepthwiseSeparableConvModule(in_channels=self.in_channels[-3]//4,
                                        out_channels=self.in_channels[-3],
                                        kernel_size=3,
                                        padding=1),    

            DepthwiseSeparableConvModule(in_channels=self.in_channels[-3],
                                        out_channels=self.in_channels[-3]//4,
                                        kernel_size=3,
                                        padding=1),

            ContextBlock(in_channels = self.in_channels[-3]//4, ratio=0.4)
        )        
        
        # 256 -> 256
        self.context_1 = nn.Sequential(
            DepthwiseSeparableConvModule(in_channels=self.in_channels[-4]
                                        , out_channels=self.in_channels[-4]//4,
                                        kernel_size=3,
                                        padding=1), 

            DepthwiseSeparableConvModule(in_channels=self.in_channels[-4]//4
                                        , out_channels=self.in_channels[-4],
                                        kernel_size=3,
                                        padding=1),   

            DepthwiseSeparableConvModule(in_channels=self.in_channels[-4]
                                        , out_channels=self.in_channels[-4]//4,
                                        kernel_size=3,
                                        padding=1),

            ContextBlock(in_channels = self.in_channels[-4]//4, ratio=0.4)
        )




        self.attention_layer = nn.Sequential(
            ConvModule(
                self.in_channels[-1],
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ISA_atten(self.channels//2, self.channels, self.channels),
            ConvModule(
                        self.channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg), 
        )
        self.conv_out = ConvModule(self.channels*2, 
                                    self.channels, 
                                    1, 
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)



    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # x = self._transform_inputs(inputs)
        psp_outs = [inputs]
        psp_outs.extend(self.psp_modules(inputs))
        psp_outs = torch.cat(psp_outs, dim=1)
        feats = self.bottleneck(psp_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""

        output = self._forward_feature(inputs[-1])
        # 32 x 32 
        output = self.conv_out(torch.cat([output, self.attention_layer(inputs[-1])], dim=1))
        # 128 x 128
        o1 = self.context_1(inputs[-4])
        o2 = torch.nn.functional.interpolate(self.context_2(inputs[-3]), scale_factor=2)
        o3 = torch.nn.functional.interpolate(output, scale_factor=2)
        o4 = torch.nn.functional.interpolate(self.context_3(inputs[-2]), scale_factor=2)
        o1 = resize(o1, size=o2.size()[2:],mode='bilinear',)
        output2 = torch.cat([o1, o2, o3, o4], dim=1)
        output = self.cls_seg(self.fusion(output2))
        return output
