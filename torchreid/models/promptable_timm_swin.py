# from collections import OrderedDict

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm

# from timm.layers import PatchEmbed

# from torchreid.models.promptable_transformer_backbone import PromptableTransformerBackbone


# class SwinTransformer(PromptableTransformerBackbone):
#     def __init__(self, name, pretrained_model, config, num_classes, img_size, in_chans_masks, enable_fpn, *args, **kwargs):
#         # timm 모델 생성
#         model = timm.create_model(name,
#                                   in_chans=3,
#                                   pretrained=True,
#                                   num_classes=num_classes,
#                                   global_pool='',
#                                   img_size=img_size,
#                                   )
#         print(model.default_cfg)
        
#         # 패치 임베딩 설정
#         patch_embed_size = model.patch_embed.grid_size
#         masks_patch_embed = PatchEmbed(
#             in_chans=in_chans_masks,
#             img_size=img_size,
#             patch_size=model.patch_embed.patch_size,
#             embed_dim=model.embed_dim,
#             norm_layer=model.patch_embed.norm.__class__ if not isinstance(model.patch_embed.norm, nn.Identity) else None,
#             output_fmt='NHWC',
#         )
        
#         # FPN 설정
#         self.enable_fpn = enable_fpn
#         spatial_feature_depth_per_layer = np.array([inf["num_chs"] for inf in model.feature_info])
        
#         if self.enable_fpn:
#             spatial_feature_depth = spatial_feature_depth_per_layer.sum()
#             spatial_feature_shape = [int(img_size[0] / model.feature_info[0]['reduction']),
#                                     int(img_size[1] / model.feature_info[0]['reduction']),
#                                     spatial_feature_depth]
#         else:
#             spatial_feature_depth = model.feature_info[-1]['num_chs']
#             spatial_feature_shape = [int(img_size[0] / model.feature_info[-1]['reduction']),
#                                     int(img_size[1] / model.feature_info[-1]['reduction']),
#                                     spatial_feature_depth]
        
#         # 부모 클래스 초기화
#         super().__init__(model.patch_embed,
#                          masks_patch_embed,
#                          patch_embed_size,
#                          config=config,
#                          patch_embed_dim=model.embed_dim,
#                          feature_dim=model.num_features,
#                          *args,
#                          **kwargs
#                          )
        
#         # 모델 및 관련 속성 설정
#         self.base = model
#         self.spatial_feature_depth_per_layer = spatial_feature_depth_per_layer
#         self.spatial_feature_depth = spatial_feature_depth
#         self.spatial_feature_shape = spatial_feature_shape
#         self.norm = nn.LayerNorm(self.spatial_feature_shape[-1])

#     def forward(self, images, prompt_masks=None, keypoints_xyc=None, cam_label=None, view_label=None, text_descriptions=None, **kwargs):
#         # 이미지 패치 임베딩
#         features = self.patch_embed(images)
        
#         # 카메라 임베딩 적용 (있는 경우)
#         if cam_label is not None or view_label is not None:
#             features = self._cam_embed(features, cam_label, view_label)
        
#         features_per_stage = OrderedDict()
        
#         # 텍스트가 활성화된 경우에만 fusion_layer 설정 사용, 그렇지 않으면 원본 방식 (i == 0) 사용
#         if hasattr(self, 'config') and hasattr(self.config, 'text_prompt_enabled') and self.config.text_prompt_enabled and text_descriptions is not None:
#             # 텍스트가 활성화된 경우 fusion_layer 설정 사용
#             fusion_layer = getattr(self.config, 'fusion_layer', 0)
#             use_fusion_layer = True
#         else:
#             # 텍스트가 비활성화된 경우 원본 방식 사용 (첫 번째 레이어에서만)
#             fusion_layer = 0
#             use_fusion_layer = False
        
#         # 각 레이어 통과
#         for i, layer in enumerate(self.base.layers):
#             if use_fusion_layer:
#                 # 텍스트가 활성화된 경우: fusion_layer 설정 또는 모든 레이어에 prompt 적용
#                 if i == fusion_layer or self.pose_encoding_all_layers:
#                     features = self._mask_embed(features, prompt_masks, images.shape[-2:], text_descriptions=text_descriptions)
#             else:
#                 # 텍스트가 비활성화된 경우: 원본 방식 (첫 번째 레이어 또는 모든 레이어)
#                 if i == 0 or self.pose_encoding_all_layers:
#                     features = self._mask_embed(features, prompt_masks, images.shape[-2:])
            
#             features = layer(features)
#             features_per_stage[i] = features.permute(0, 3, 1, 2)

#         # FPN 사용 여부에 따른 출력 설정
#         if self.enable_fpn:
#             features = features_per_stage
#         else:
#             features = features_per_stage[list(features_per_stage.keys())[-1]]  # 마지막 레이어 출력
#             features = self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            
#         return features


# def timm_swin(
#         name="",
#         config=None,
#         cam_num=0,
#         view=0,
#         num_classes=0,
#         enable_fpn=True,
#         **kwargs,
# ):
#     no_background_token = config.model.promptable_trans.no_background_token
#     use_negative_keypoints = config.model.kpr.keypoints.use_negative_keypoints
#     in_chans_masks = config.model.kpr.masks.prompt_parts_num
    
#     if not no_background_token:
#         in_chans_masks += 1
#     if use_negative_keypoints:
#         in_chans_masks += 1
    
#     # 디버깅을 위한 정보 출력
#     # print(f"Creating SwinTransformer with in_chans_masks={in_chans_masks}")
    
#     # 텍스트가 활성화된 경우에만 채널 수를 8로 조정
#     text_enabled = getattr(config, 'text_prompt', None) and getattr(config.text_prompt, 'enabled', False)
#     if text_enabled:
#         # 채널 수가 8개가 아닌 경우 경고하고 조정
#         if in_chans_masks != 8:
#             print(f"Warning: For text prompts, masks_patch_embed expects 8 channels, but calculated {in_chans_masks}. Adjusting to 8.")
#             in_chans_masks = 8
#     else:
#         # 텍스트가 비활성화된 경우 원본 채널 수 유지
#         print(f"Text prompts disabled. Using calculated channel count: {in_chans_masks}")
    
#     # kwargs에 full_config 추가
#     kwargs['full_config'] = config
    
#     # 모델 생성
#     model = SwinTransformer(
#         name=name,
#         pretrained_model="",
#         config=config.model.promptable_trans,
#         num_classes=0,
#         use_negative_keypoints=config.model.kpr.keypoints.use_negative_keypoints,
#         img_size=[config.data.height, config.data.width],
#         in_chans_masks=in_chans_masks,
#         camera=cam_num if config.model.transreid.sie_camera else 0,
#         view=view if config.model.transreid.sie_view else 0,
#         sie_xishu=config.model.transreid.sie_coe,
#         masks_prompting=config.model.promptable_trans.masks_prompting,
#         disable_inference_prompting=config.model.promptable_trans.disable_inference_prompting,
#         prompt_parts_num=config.model.kpr.masks.prompt_parts_num,
#         enable_fpn=enable_fpn,
#         **kwargs,
#     )
#     return model


# swin_timm_models = {
#     "swin_base_patch4_window12_384.ms_in1k": timm_swin,
#     "swin_large_patch4_window12_384.ms_in22k_ft_in1k": timm_swin,
#     'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k': timm_swin,
#     'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k': timm_swin,
#     'swinv2_base_window8_256.ms_in1k': timm_swin,
#     'swinv2_base_window16_256.ms_in1k': timm_swin,
#     'swinv2_base_window12_192.ms_in22k': timm_swin,
#     'swin_base_patch4_window7_224.ms_in22k_ft_in1k': timm_swin,
# }
from collections import OrderedDict

import numpy as np
import torch.nn as nn
import timm

from timm.layers import PatchEmbed

from torchreid.models.promptable_transformer_backbone import PromptableTransformerBackbone


class SwinTransformer(PromptableTransformerBackbone):
    def __init__(self, name, num_classes, img_size, in_chans_masks, enable_fpn, *args, **kwargs):
        model = timm.create_model(name,
                                  pretrained=True,
                                  num_classes=num_classes,
                                  global_pool='',
                                  img_size=img_size,
                                  )
        print(model.default_cfg)
        patch_embed_size = model.patch_embed.grid_size
        masks_patch_embed = PatchEmbed(
            in_chans=in_chans_masks,
            img_size=img_size,
            patch_size=model.patch_embed.patch_size,
            embed_dim=model.embed_dim,
            norm_layer=model.patch_embed.norm.__class__ if not isinstance(model.patch_embed.norm, nn.Identity) else None,
            output_fmt='NHWC',
        )
        self.enable_fpn = enable_fpn
        self.spatial_feature_depth_per_layer = np.array([inf["num_chs"] for inf in model.feature_info])
        if self.enable_fpn:
            self.spatial_feature_depth = self.spatial_feature_depth_per_layer.sum()
            self.spatial_feature_shape = [int(img_size[0] / model.feature_info[0]['reduction']),
                                          int(img_size[1] / model.feature_info[0]['reduction']),
                                          self.spatial_feature_depth]
        else:
            self.spatial_feature_depth = model.feature_info[-1]['num_chs']
            self.spatial_feature_shape = [int(img_size[0] / model.feature_info[-1]['reduction']),
                                          int(img_size[1] / model.feature_info[-1]['reduction']),
                                          self.spatial_feature_depth]
        super().__init__(model.patch_embed,
                         masks_patch_embed,
                         patch_embed_size,
                         patch_embed_dim=model.embed_dim,
                         feature_dim=model.num_features,
                         *args,
                         **kwargs
                         )
        self.model = model
        self.norm = nn.LayerNorm(self.spatial_feature_shape[-1])
        # feature pyramid network to build high resolution semantic feature maps from multiple stages:

    def forward(self, images, prompt_masks=None, keypoints_xyc=None, cam_label=None, view_label=None, **kwargs):
        features = self.model.patch_embed(images)
        if cam_label is not None or view_label is not None:
            features = self._cam_embed(features, cam_label, view_label)
        features_per_stage = OrderedDict()
        for i, layer in enumerate(self.model.layers):
            if i == 0 or self.pose_encoding_all_layers:  # TODO make it work in other scenarios/configs
                features = self._mask_embed(features, prompt_masks, images.shape[-2::])
            features = layer(features)
            features_per_stage[i] = features.permute(0, 3, 1, 2)

        if self.enable_fpn:
            features = features_per_stage
        else:
            features = features_per_stage[list(features_per_stage.keys())[-1]]  # last layer output
            features = self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # TODO apply it also after FPN?
        return features



def timm_swin(
        name="",
        config=None,
        cam_num=0,
        view=0,
        num_classes=0,
        enable_fpn=True,
        **kwargs,
):
    no_background_token = config.model.promptable_trans.no_background_token
    use_negative_keypoints = config.model.kpr.keypoints.use_negative_keypoints
    in_chans_masks = config.model.kpr.masks.prompt_parts_num
    if not no_background_token:
        in_chans_masks += 1
    if use_negative_keypoints:
        in_chans_masks += 1
    model = SwinTransformer(
        name=name,
        pretrained_model="",
        config=config.model.promptable_trans,
        num_classes=0,
        use_negative_keypoints=config.model.kpr.keypoints.use_negative_keypoints,
        img_size=[config.data.height, config.data.width],
        in_chans_masks=in_chans_masks,
        camera=cam_num if config.model.transreid.sie_camera else 0,
        view=view if config.model.transreid.sie_view else 0,
        sie_xishu=config.model.transreid.sie_coe,
        masks_prompting=config.model.promptable_trans.masks_prompting,
        disable_inference_prompting=config.model.promptable_trans.disable_inference_prompting,
        prompt_parts_num=config.model.kpr.masks.prompt_parts_num,
        enable_fpn=enable_fpn,
        **kwargs,
    )
    return model


swin_timm_models = {
    "swin_base_patch4_window12_384.ms_in1k": timm_swin,
    "swin_large_patch4_window12_384.ms_in22k_ft_in1k": timm_swin,
    'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k': timm_swin,
    'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k': timm_swin,
    'swinv2_base_window8_256.ms_in1k': timm_swin,
    'swinv2_base_window16_256.ms_in1k': timm_swin,
    'swinv2_base_window12_192.ms_in22k': timm_swin,
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k': timm_swin,
}
