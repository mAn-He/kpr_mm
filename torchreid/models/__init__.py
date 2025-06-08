from __future__ import absolute_import
from .hrnet import hrnet32
from .kpr import KPR
# from .pcb import * # Removed
# from .mlfn import * # Removed
# from .hacnn import * # Removed
# from .osnet import * # Removed
from .promptable_solider import solider_models
# from .pvpm import pose_resnet50_256_p4, pose_resnet50_256_p6, pose_resnet50_256_p6_pscore_reg, \
#     pose_resnet50_256_p4_pscore_reg # Removed
# from .resnet_fastreid import build_resnet_backbone, fastreid_resnet, fastreid_resnet_ibn, fastreid_resnet_nl, \
#     fastreid_resnet_ibn_nl # Removed
from .sam import *
# from .senet import * # Removed
# from .mudeep import * # Removed
# from .nasnet import * # Removed
from .resnet import *
# from .densenet import * # Removed
from .solider import transreid_solider
from .promptable_timm_swin import swin_timm_models
from .promptable_timm_vit import vit_timm_models
from .transreid import transreid
from .promptable_vit import vit
# from .xception import * # Removed
# from .osnet_ain import * # Removed
# from .resnetmid import * # Removed
# from .shufflenet import * # Removed
# from .squeezenet import * # Removed
# from .inceptionv4 import * # Removed
# from .mobilenetv2 import * # Removed
from .resnet_ibn_a import *
from .resnet_ibn_b import *
# from .shufflenetv2 import * # Removed
# from .inceptionresnetv2 import * # Removed

__model_factory = {
    # image classification models
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'resnet50_fc512': resnet50_fc512,
    # 'se_resnet50': se_resnet50, # Removed
    # 'se_resnet50_fc512': se_resnet50_fc512, # Removed
    # 'se_resnet101': se_resnet101, # Removed
    # 'se_resnext50_32x4d': se_resnext50_32x4d, # Removed
    # 'se_resnext101_32x4d': se_resnext101_32x4d, # Removed
    # 'densenet121': densenet121, # Removed
    # 'densenet169': densenet169, # Removed
    # 'densenet201': densenet201, # Removed
    # 'densenet161': densenet161, # Removed
    # 'densenet121_fc512': densenet121_fc512, # Removed
    # 'inceptionresnetv2': inceptionresnetv2, # Removed
    # 'inceptionv4': inceptionv4, # Removed
    # 'xception': xception, # Removed
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet50_ibn_b': resnet50_ibn_b,
    # lightweight models
    # 'nasnsetmobile': nasnetamobile, # Removed
    # 'mobilenetv2_x1_0': mobilenetv2_x1_0, # Removed
    # 'mobilenetv2_x1_4': mobilenetv2_x1_4, # Removed
    # 'shufflenet': shufflenet, # Removed
    # 'squeezenet1_0': squeezenet1_0, # Removed
    # 'squeezenet1_0_fc512': squeezenet1_0_fc512, # Removed
    # 'squeezenet1_1': squeezenet1_1, # Removed
    # 'shufflenet_v2_x0_5': shufflenet_v2_x0_5, # Removed
    # 'shufflenet_v2_x1_0': shufflenet_v2_x1_0, # Removed
    # 'shufflenet_v2_x1_5': shufflenet_v2_x1_5, # Removed
    # 'shufflenet_v2_x2_0': shufflenet_v2_x2_0, # Removed
    # reid-specific models
    # 'mudeep': MuDeep, # Removed
    # 'resnet50mid': resnet50mid, # Removed
    # 'hacnn': HACNN, # Removed
    # 'pcb_p6': pcb_p6, # Removed
    # 'pcb_p4': pcb_p4, # Removed
    # 'mlfn': mlfn, # Removed
    # 'osnet_x1_0': osnet_x1_0, # Removed
    # 'osnet_x0_75': osnet_x0_75, # Removed
    # 'osnet_x0_5': osnet_x0_5, # Removed
    # 'osnet_x0_25': osnet_x0_25, # Removed
    # 'osnet_ibn_x1_0': osnet_ibn_x1_0, # Removed
    # 'osnet_ain_x1_0': osnet_ain_x1_0, # Removed
    # 'pose_p4': pose_resnet50_256_p4, # Removed
    # 'pose_p6': pose_resnet50_256_p6, # Removed
    # 'pose_p6s': pose_resnet50_256_p6_pscore_reg, # Removed
    # 'pose_p4s': pose_resnet50_256_p4_pscore_reg, # Removed
    'hrnet32': hrnet32,
    'kpr': KPR,
    # 'fastreid_resnet': fastreid_resnet, # Removed
    # 'fastreid_resnet_ibn': fastreid_resnet_ibn, # Removed
    # 'fastreid_resnet_nl': fastreid_resnet_nl, # Removed
    # 'fastreid_resnet_ibn_nl': fastreid_resnet_ibn_nl, # Removed
    'sam_vit_h': sam_vit_h,
    'sam_vit_l': sam_vit_l,
    'sam_vit_b': sam_vit_b,
    'transreid': transreid,
    'transreid_solider': transreid_solider,
    'vit': vit,
    **swin_timm_models,
    **vit_timm_models,
    **solider_models,
}


def show_avai_models():
    """Displays available models.

    Examples::
        >>> from torchreid import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


def build_model(
    name, num_classes=1, loss='softmax', pretrained=True, use_gpu=True, **kwargs
):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    kwargs.pop("name", None)
    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        name=name,
        **kwargs
    )
