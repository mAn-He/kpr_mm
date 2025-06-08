# from collections import OrderedDict

# import torch
# import math
# from torch import nn as nn
# from torch.nn import functional as F
# import clip

# from torchreid.models.kpr import AfterPoolingDimReduceLayer

# class PromptableTransformerBackbone(nn.Module):
#     """ class to be inherited by all promptable transformer backbones.
#     It defines how prompt should be tokenized (i.e. the implementation of the prompt tokenizer).
#     It also defines how camera information should be embedded similar to Transreid.
#     """
#     def __init__(self,
#                  patch_embed,
#                  masks_patch_embed,
#                  patch_embed_size,
#                  config,
#                  patch_embed_dim,
#                  feature_dim,
#                  full_config=None,
#                  use_negative_keypoints=False,
#                  camera=0,
#                  view=0,
#                  sie_xishu =1.0,
#                  masks_prompting=False,
#                  disable_inference_prompting=False,
#                  prompt_parts_num=0,
#                  **kwargs,
#                  ):
#         super().__init__()
#         self.full_config = full_config
#         # standard params
#         self.feature_dim = self.num_features = feature_dim  # num_features for consistency with other models
#         self.patch_embed_dim = patch_embed_dim

#         # prompt related params
#         self.masks_prompting = masks_prompting
#         self.disable_inference_prompting = disable_inference_prompting
#         self.prompt_parts_num = prompt_parts_num
#         self.pose_encoding_strategy = config.pose_encoding_strategy
#         self.pose_encoding_all_layers = config.pose_encoding_all_layers
#         self.no_background_token = config.no_background_token
#         self.use_negative_keypoints = use_negative_keypoints

#         # patch embedding for image and prompt
#         self.patch_embed_size = patch_embed_size
#         self.patch_embed = patch_embed
#         self.masks_patch_embed = masks_patch_embed
#         self.num_patches = self.patch_embed_size[0] * self.patch_embed_size[1]

#         # token for camera and view
#         self.cam_num = camera
#         self.view_num = view
#         self.sie_xishu = sie_xishu

#         # Initialize SIE Embedding
#         if camera > 1 and view > 1:
#             self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, self.patch_embed_dim))
#             trunc_normal_(self.sie_embed, std=.02)
#             print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
#             print('using SIE_Lambda is : {}'.format(sie_xishu))
#         elif camera > 1:
#             self.sie_embed = nn.Parameter(torch.zeros(camera, 1, self.patch_embed_dim))
#             trunc_normal_(self.sie_embed, std=.02)
#             print('camera number is : {}'.format(camera))
#             print('using SIE_Lambda is : {}'.format(sie_xishu))
#         elif view > 1:
#             self.sie_embed = nn.Parameter(torch.zeros(view, 1, self.patch_embed_dim))
#             trunc_normal_(self.sie_embed, std=.02)
#             print('viewpoint number is : {}'.format(view))
#             print('using SIE_Lambda is : {}'.format(sie_xishu))
#         else:
#             self.sie_embed = None

#         # token for parts
#         self.num_part_tokens = self.prompt_parts_num + 1
#         if self.use_negative_keypoints:
#             self.num_part_tokens += 1
#         self.parts_embed = nn.Parameter(torch.zeros(self.num_part_tokens, 1, self.patch_embed_dim))  # +1 for background
#         self.num_layers = 4  # FIXME
#         if self.pose_encoding_all_layers:
#             self.parts_embed_dim_upscales = nn.ModuleDict({str(self.patch_embed_dim * 2 ** i) : AfterPoolingDimReduceLayer(self.patch_embed_dim, self.patch_embed_dim * 2 ** i) for i in range(self.num_layers-1)})

#         # init tokens
#         trunc_normal_(self.parts_embed, std=.02)

#         # [NEW] Initialize CLIP model for text encoding
#         self.text_prompt_enabled = False
#         self.clip_model = None
#         self.clip_preprocess = None # CLIP's own image preprocessor, not used here for text
#         self.text_feature_proj = None # Optional: for projecting text features to match image feature dim

#         if self.full_config and self.full_config.model.text_prompt.enabled:
#             self.text_prompt_enabled = True
#             clip_model_name = self.full_config.model.text_prompt.clip_model_name
#             try:
#                 # Determine device for CLIP model
#                 device = torch.device("cuda" if torch.cuda.is_available() and self.full_config.use_gpu else "cpu")
#                 print(f"Loading CLIP model {clip_model_name} on device: {device}")
#                 self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
#                 self.clip_model.eval() # Set to eval mode as we only use it for encoding

#                 # Optional: Project text features to image patch_embed_dim if they differ
#                 # CLIP text feature dim depends on the model (e.g., 512 for ViT-B/32, 768 for ViT-L/14)
#                 text_feature_dim = self.clip_model.text_projection.shape[-1] # Or other ways to get text feature dim
#                 if text_feature_dim != self.patch_embed_dim:
#                     print(f"Projecting CLIP text features from {text_feature_dim} to {self.patch_embed_dim}")
#                     self.text_feature_proj = nn.Linear(text_feature_dim, self.patch_embed_dim).to(device)
#                 else:
#                      self.text_feature_proj = nn.Identity().to(device)


#             except Exception as e:
#                 print(f"Error loading CLIP model: {e}. Disabling text prompts.")
#                 self.text_prompt_enabled = False
#                 self.clip_model = None

#     def _cam_embed(self, images, cam_label, view_label):
#         reshape = False
#         if len(images.shape) == 4:
#             b, h, w, c = images.shape
#             images = images.view(b, h * w, c)
#             reshape = True
#         if self.cam_num > 0 and self.view_num > 0:
#             images = images + self.sie_xishu * self.sie_embed[cam_label * self.view_num + view_label]
#         elif self.cam_num > 0:
#             images = images + self.sie_xishu * self.sie_embed[cam_label]
#         elif self.view_num > 0:
#             images = images + self.sie_xishu * self.sie_embed[view_label]
#         else:
#             images = images
#         if reshape:
#             images = images.view(b, h, w, c)
#         return images

#     """The Prompt Tokenizer, to tokenize the input keypoint prompt information and add it to images tokens.
#     Here, keypoints prompts in the (x, y, c) format are already pre-processed (see 'torchreid/data/datasets/dataset.py -> ImageDataset.getitem()') 
#     and turned into dense heatmaps of shape (K+2, H, W) where K is the number of parts, and K+2 include the negative keypoints and the background, and H, W are the height and width of the image.
#     'prompt_masks' is therefore a tensor of shape (B, K+2, H, W) where B is the batch size."""
#     def _mask_embed(self, image_features, prompt_masks, input_size, text_descriptions=None):
#         """The Prompt Tokenizer, to tokenize the input keypoint prompt information and add it to images tokens."""
#         # 이미지 피처의 원래 형태 저장 (디버깅용)
#         original_shape = image_features.shape
#         # print(f"_mask_embed - image_features original shape: {original_shape}")
        
#         if self.masks_prompting:
#             if prompt_masks is not None and prompt_masks.shape[2:] != input_size:
#                 prompt_masks = F.interpolate(
#                     prompt_masks,
#                     size=input_size,
#                     mode="bilinear",
#                     align_corners=True
#                 )
                
#             if self.disable_inference_prompting or prompt_masks is None:
#                 prompt_masks = torch.zeros([image_features.shape[0], self.num_part_tokens, input_size[0], input_size[1]], device=image_features.device)
#                 if not self.no_background_token:
#                     prompt_masks[:, 0] = 1.
                    
#             prompt_masks = prompt_masks.type(image_features.dtype)
            
#             if self.pose_encoding_strategy == 'embed_heatmaps_patches':
#                 prompt_masks.requires_grad = False
#                 if self.no_background_token:
#                     prompt_masks = prompt_masks[:, 1:]
                
#                 # 텍스트가 활성화된 경우에만 채널 수 조정
#                 if self.text_prompt_enabled and text_descriptions is not None:
#                     # 채널 수 조정 (8채널로 맞추기)
#                     if prompt_masks.shape[1] != 8:
#                         if prompt_masks.shape[1] < 8:
#                             # 부족한 채널 추가
#                             padding = torch.zeros(
#                                 (prompt_masks.shape[0], 8 - prompt_masks.shape[1], *prompt_masks.shape[2:]),
#                                 device=prompt_masks.device,
#                                 dtype=prompt_masks.dtype
#                             )
#                             prompt_masks = torch.cat([prompt_masks, padding], dim=1)
#                         else:
#                             # 초과 채널 제거
#                             prompt_masks = prompt_masks[:, :8]
                
#                 # 프롬프트 마스크 임베딩 (원래 코드와 동일하게 유지)
#                 part_tokens = self.masks_patch_embed(prompt_masks)
#                 part_tokens = part_tokens[0] if isinstance(part_tokens, tuple) else part_tokens
                
#                 # 디버깅용 로그
#                 # print(f"_mask_embed - part_tokens shape after embedding: {part_tokens.shape}")
                
#                 # 텍스트 설명 처리 (CLIP 모델이 있고 텍스트 설명이 제공된 경우)
#                 if self.text_prompt_enabled and text_descriptions is not None and self.clip_model is not None:
#                     try:
#                         # 텍스트 설명을 CLIP 모델로 인코딩
#                         with torch.no_grad():  # 추론 모드
#                             text_tokens = clip.tokenize(text_descriptions).to(image_features.device)
#                             text_features = self.clip_model.encode_text(text_tokens)
                        
#                         # 텍스트 피처 정규화
#                         norm = text_features.norm(dim=-1, keepdim=True)
#                         norm = torch.clamp(norm, min=1e-8)  # 0으로 나누는 것 방지
#                         text_features = text_features / norm
                        
#                         # 필요한 경우 프로젝션 적용
#                         if self.text_feature_proj is not None:
#                             text_features = self.text_feature_proj(text_features)
                        
#                         # 중요: 텍스트 피처를 이미지 피처 형태에 맞게 조정
#                         # part_tokens의 형태와 정확히 일치하도록 조정
#                         text_features = text_features.view(text_features.shape[0], 1, 1, -1)
#                         text_features = text_features.expand_as(part_tokens)
                        
#                         # 텍스트 피처와 파트 토큰 결합 (요소별 합)
#                         # 중요: part_tokens의 형태를 유지하기 위해 할당 연산 사용
#                         part_tokens = part_tokens + 0.5 * text_features  # 가중치 조정 가능 (0.5는 예시)
                        
#                         # 디버깅용 로그
#                         # print(f"_mask_embed - part_tokens shape after adding text: {part_tokens.shape}")
#                     except Exception as e:
#                         print(f"Error processing text descriptions: {e}")
                
#             elif self.pose_encoding_strategy == 'spatialize_part_tokens':
#                 parts_embed = self.parts_embed
#                 if parts_embed.shape[-1] != image_features.shape[-1]:
#                     parts_embed = self.parts_embed_dim_upscales[str(image_features.shape[-1])](parts_embed)
#                 prompt_masks.requires_grad = False
#                 parts_segmentation_map = prompt_masks.argmax(dim=1)
#                 part_tokens = parts_embed[parts_segmentation_map].squeeze(-2)
#                 if self.no_background_token:
#                     part_tokens[parts_segmentation_map == 0] = 0
#                 if len(part_tokens.shape) != len(image_features.shape):
#                     part_tokens = part_tokens.flatten(start_dim=1, end_dim=2)
                
#                 # 디버깅용 로그
#                 # print(f"_mask_embed - part_tokens shape (spatialize): {part_tokens.shape}")
                
#                 # 텍스트 설명 처리 (CLIP 모델이 있고 텍스트 설명이 제공된 경우)
#                 if self.text_prompt_enabled and text_descriptions is not None and self.clip_model is not None:
#                     try:
#                         # 텍스트 설명을 CLIP 모델로 인코딩
#                         with torch.no_grad():
#                             text_tokens = clip.tokenize(text_descriptions).to(image_features.device)
#                             text_features = self.clip_model.encode_text(text_tokens)
                        
#                         # 텍스트 피처 정규화
#                         norm = text_features.norm(dim=-1, keepdim=True)
#                         norm = torch.clamp(norm, min=1e-8)  # 0으로 나누는 것 방지
#                         text_features = text_features / norm
                        
#                         # 필요한 경우 프로젝션 적용
#                         if self.text_feature_proj is not None:
#                             text_features = self.text_feature_proj(text_features)
                        
#                         # 중요: 텍스트 피처를 part_tokens의 형태에 정확히 맞추기
#                         if len(part_tokens.shape) == 4:  # [B, H, W, C]
#                             text_features = text_features.view(text_features.shape[0], 1, 1, -1)
#                             text_features = text_features.expand_as(part_tokens)
#                         elif len(part_tokens.shape) == 3:  # [B, N, C]
#                             text_features = text_features.view(text_features.shape[0], 1, -1)
#                             text_features = text_features.expand_as(part_tokens)
#                         elif len(part_tokens.shape) == 2:  # [B, C]
#                             text_features = text_features.expand_as(part_tokens)
                        
#                         # 텍스트 피처와 파트 토큰 결합 (요소별 합)
#                         # 중요: part_tokens의 형태를 유지하기 위해 할당 연산 사용
#                         part_tokens = part_tokens + 0.5 * text_features
                        
#                         # 디버깅용 로그
#                         # print(f"_mask_embed - part_tokens shape after adding text: {part_tokens.shape}")
#                     except Exception as e:
#                         print(f"Error processing text descriptions: {e}")
#             else:
#                 raise NotImplementedError
            
#             # 텍스트가 활성화된 경우에만 고급 에러 처리 사용
#             if self.text_prompt_enabled and text_descriptions is not None:
#                 # 형태 확인 (디버깅용)
#                 # print(f"_mask_embed - Before addition: image_features={image_features.shape}, part_tokens={part_tokens.shape}")
                
#                 # 형태가 일치하는지 확인 후 element-wise 합 수행
#                 if image_features.shape == part_tokens.shape:
#                     # 원래 코드와 동일하게 유지 (in-place 연산)
#                     image_features += part_tokens
#                     # print("_mask_embed - Element-wise addition successful")
#                 else:
#                     print(f"_mask_embed - Warning: Shape mismatch! Adjusting part_tokens to match image_features")
#                     # 이미지 피처 형태에 맞게 part_tokens 조정 시도
#                     try:
#                         # 기존 형태로 리사이징
#                         if len(image_features.shape) == 4 and len(part_tokens.shape) == 4:  # [B, H, W, C]
#                             if image_features.shape[1:3] != part_tokens.shape[1:3]:
#                                 # 높이, 너비 불일치 - 공간 차원만 맞추기
#                                 part_tokens = F.interpolate(
#                                     part_tokens.permute(0, 3, 1, 2),  # [B, H, W, C] -> [B, C, H, W]
#                                     size=image_features.shape[1:3],
#                                     mode="bilinear",
#                                     align_corners=True
#                                 ).permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
                        
#                         # element-wise 합 시도
#                         image_features += part_tokens
#                         # print(f"_mask_embed - Element-wise addition after adjustment successful")
#                     except Exception as e:
#                         print(f"_mask_embed - Error during adjustment: {e}")
#                         print("_mask_embed - Returning original image_features")
#             else:
#                 # 텍스트가 비활성화된 경우 원본과 동일한 단순한 융합 사용
#                 image_features += part_tokens
        
#         # 텍스트가 활성화된 경우에만 최종 형태 확인
#         if self.text_prompt_enabled and text_descriptions is not None:
#             if original_shape != image_features.shape:
#                 print(f"_mask_embed - Warning: image_features shape changed from {original_shape} to {image_features.shape}")
        
#         return image_features

#     def _combine_layers(self, features, layers, prompt_masks, text_descriptions=None):
#         features_per_stage = OrderedDict()
#         for i, layer in enumerate(layers):
#             features_size = features.shape[-2::]
#             features = layer(features)
#             if self.pose_encoding_all_layers:  # TODO make it work in other scenarios/configs
#                 features = self._mask_embed(features, prompt_masks, features_size, text_descriptions=text_descriptions if i==0 else None)
#             features_per_stage[i] = features.permute(0, 3, 1, 2)
#         return features_per_stage


# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     # type: (Tensor, float, float, float, float) -> Tensor
#     r"""Fills the input Tensor with values drawn from a truncated
#     normal distribution. The values are effectively drawn from the
#     normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
#     with values outside :math:`[a, b]` redrawn until they are within
#     the bounds. The method used for generating the random values works
#     best when :math:`a \leq \text{mean} \leq b`.
#     Args:
#         tensor: an n-dimensional `torch.Tensor`
#         mean: the mean of the normal distribution
#         std: the standard deviation of the normal distribution
#         a: the minimum cutoff value
#         b: the maximum cutoff value
#     Examples:
#         >>> w = torch.empty(3, 5)
#         >>> nn.init.trunc_normal_(w)
#     """
#     return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# def _no_grad_trunc_normal_(tensor, mean, std, a, b):
#     # Cut & paste from PyTorch official master until it's in a few official releases - RW
#     # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
#     def norm_cdf(x):
#         # Computes standard normal cumulative distribution function
#         return (1. + math.erf(x / math.sqrt(2.))) / 2.

#     if (mean < a - 2 * std) or (mean > b + 2 * std):
#         print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
#                       "The distribution of values may be incorrect.",)

#     with torch.no_grad():
#         # Values are generated by using a truncated uniform distribution and
#         # then using the inverse CDF for the normal distribution.
#         # Get upper and lower cdf values
#         l = norm_cdf((a - mean) / std)
#         u = norm_cdf((b - mean) / std)

#         # Uniformly fill tensor with values from [l, u], then translate to
#         # [2l-1, 2u-1].
#         tensor.uniform_(2 * l - 1, 2 * u - 1)

#         # Use inverse cdf transform for normal distribution to get truncated
#         # standard normal
#         tensor.erfinv_()

#         # Transform to proper mean, std
#         tensor.mul_(std * math.sqrt(2.))
#         tensor.add_(mean)

#         # Clamp to ensure it's in the proper range
#         tensor.clamp_(min=a, max=b)
#         return tensor
from collections import OrderedDict

import torch
import math
from torch import nn as nn
from torch.nn import functional as F

from torchreid.models.kpr import AfterPoolingDimReduceLayer

class PromptableTransformerBackbone(nn.Module):
    """ class to be inherited by all promptable transformer backbones.
    It defines how prompt should be tokenized (i.e. the implementation of the prompt tokenizer).
    It also defines how camera information should be embedded similar to Transreid.
    """
    def __init__(self,
                 patch_embed,
                 masks_patch_embed,
                 patch_embed_size,
                 config,
                 patch_embed_dim,
                 feature_dim,
                 use_negative_keypoints=False,
                 camera=0,
                 view=0,
                 sie_xishu =1.0,
                 masks_prompting=False,
                 disable_inference_prompting=False,
                 prompt_parts_num=0,
                 **kwargs,
                 ):
        super().__init__()

        # standard params
        self.feature_dim = self.num_features = feature_dim  # num_features for consistency with other models
        self.patch_embed_dim = patch_embed_dim

        # prompt related params
        self.masks_prompting = masks_prompting
        self.disable_inference_prompting = disable_inference_prompting
        self.prompt_parts_num = prompt_parts_num
        self.pose_encoding_strategy = config.pose_encoding_strategy
        self.pose_encoding_all_layers = config.pose_encoding_all_layers
        self.no_background_token = config.no_background_token
        self.use_negative_keypoints = use_negative_keypoints

        # patch embedding for image and prompt
        self.patch_embed_size = patch_embed_size
        self.patch_embed = patch_embed
        self.masks_patch_embed = masks_patch_embed
        self.num_patches = self.patch_embed_size[0] * self.patch_embed_size[1]

        # token for camera and view
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu

        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, self.patch_embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, self.patch_embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, self.patch_embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        else:
            self.sie_embed = None

        # token for parts
        self.num_part_tokens = self.prompt_parts_num + 1
        if self.use_negative_keypoints:
            self.num_part_tokens += 1
        self.parts_embed = nn.Parameter(torch.zeros(self.num_part_tokens, 1, self.patch_embed_dim))  # +1 for background
        self.num_layers = 4  # FIXME
        if self.pose_encoding_all_layers:
            self.parts_embed_dim_upscales = nn.ModuleDict({str(self.patch_embed_dim * 2 ** i) : AfterPoolingDimReduceLayer(self.patch_embed_dim, self.patch_embed_dim * 2 ** i) for i in range(self.num_layers-1)})

        # init tokens
        trunc_normal_(self.parts_embed, std=.02)

    def _cam_embed(self, images, cam_label, view_label):
        reshape = False
        if len(images.shape) == 4:
            b, h, w, c = images.shape
            images = images.view(b, h * w, c)
            reshape = True
        if self.cam_num > 0 and self.view_num > 0:
            images = images + self.sie_xishu * self.sie_embed[cam_label * self.view_num + view_label]
        elif self.cam_num > 0:
            images = images + self.sie_xishu * self.sie_embed[cam_label]
        elif self.view_num > 0:
            images = images + self.sie_xishu * self.sie_embed[view_label]
        else:
            images = images
        if reshape:
            images = images.view(b, h, w, c)
        return images

    """The Prompt Tokenizer, to tokenize the input keypoint prompt information and add it to images tokens.
    Here, keypoints prompts in the (x, y, c) format are already pre-processed (see 'torchreid/data/datasets/dataset.py -> ImageDataset.getitem()') 
    and turned into dense heatmaps of shape (K+2, H, W) where K is the number of parts, and K+2 include the negative keypoints and the background, and H, W are the height and width of the image.
    'prompt_masks' is therefore a tensor of shape (B, K+2, H, W) where B is the batch size."""
    def _mask_embed(self, image_features, prompt_masks, input_size):
        if self.masks_prompting:
            if prompt_masks is not None and prompt_masks.shape[2:] != input_size:
                prompt_masks = F.interpolate(
                    prompt_masks,
                    size=input_size,
                    mode="bilinear",
                    align_corners=True
                )
            if self.disable_inference_prompting or prompt_masks is None:
                prompt_masks = torch.zeros([image_features.shape[0], self.num_part_tokens, input_size[0], input_size[1]], device=image_features.device)
                if not self.no_background_token:
                    prompt_masks[:, 0] = 1.  # if the background token was enabled when training the model, the "empty"
                    # prompts the model has seen during training are prompts with 0 filled heatmaps on each channel, and 1 filled heatmap on the background channel.
                    # The model should therefore be prompted with a similar empty prompt during inference when prompts are disabled.
            prompt_masks = prompt_masks.type(image_features.dtype)
            if self.pose_encoding_strategy == 'embed_heatmaps_patches':
                prompt_masks.requires_grad = False  # should be unecessary
                if self.no_background_token:
                    prompt_masks = prompt_masks[:, 1:]  # remove background mask that was generated with the AddBackgroundMask transform
                part_tokens = self.masks_patch_embed(prompt_masks)
                part_tokens = part_tokens[0] if isinstance(part_tokens, tuple) else part_tokens
            elif self.pose_encoding_strategy == 'spatialize_part_tokens':  # TODO add another variant where token multiplied by continuous mask
                parts_embed = self.parts_embed
                if parts_embed.shape[-1] != image_features.shape[-1]:
                    parts_embed = self.parts_embed_dim_upscales[str(image_features.shape[-1])](parts_embed)
                prompt_masks.requires_grad = False
                parts_segmentation_map = prompt_masks.argmax(
                    dim=1)  # map each patch to a part index (or background)
                part_tokens = parts_embed[parts_segmentation_map].squeeze(-2)  # map each patch to a part token
                if self.no_background_token:
                    part_tokens[parts_segmentation_map == 0] = 0  # FIXME if no_background_token, make the background token a non learnable/zero vector
                    # TODO if no background token, add only part_token where necessary, with images[parts_segmentation_map] += part_tokens[parts_segmentation_map]
                if len(part_tokens.shape) != len(image_features.shape):
                    part_tokens = part_tokens.flatten(start_dim=1, end_dim=2)
            else:
                raise NotImplementedError
            image_features += part_tokens
        return image_features

    def _combine_layers(self, features, layers, prompt_masks):  # TODO remove unused
        features_per_stage = OrderedDict()
        for i, layer in enumerate(layers):
            features_size = features.shape[-2::]
            features = layer(features)
            if self.pose_encoding_all_layers:  # TODO make it work in other scenarios/configs
                self._mask_embed(features, prompt_masks, features_size)
            features_per_stage[i] = features.permute(0, 3, 1, 2)
        return features_per_stage


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
