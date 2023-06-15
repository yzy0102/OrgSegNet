# Copyright (c) OpenMMLab. All rights reserved.
from .aspp_head import ASPPHead
from .da_head import DAHead
from .dnl_head import DNLHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .isa_head import ISAHead
from .nl_head import NLHead
from .psp_head import PSPHead
from .uper_head import UPerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .orgseg_head import OrgSeg_Head

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead',  'NLHead', 
    'UPerHead', 'DepthwiseSeparableASPPHead',  'DAHead', 
    'DepthwiseSeparableFCNHead', 'FPNHead',  'DNLHead', 'OrgSeg_Head', 'DepthwiseSeparableFCNHead'
]
