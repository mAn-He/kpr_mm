from __future__ import division, absolute_import

import torch
import torch.nn as nn
from collections import OrderedDict
from torchmetrics import Accuracy

from torchreid.losses import init_part_based_triplet_loss, CrossEntropyLoss
from torchreid.losses.supconloss import SupervisedContrastiveLoss
from torchreid.utils.constants import GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS, SUPERCON
# 파일: torchreid/losses/GiLt_loss.py


class GiLtconLoss(nn.Module):
    """
    기존 part-based ReID (CE+Triplet) + SupervisedContrastiveLoss
    GiLt_loss의 안정성 유지하면서 SuperCon 기능 추가
    """
    default_losses_weights = {
        GLOBAL: {'id': 1., 'tr': 0.},
        FOREGROUND: {'id': 1., 'tr': 0.},
        CONCAT_PARTS: {'id': 1., 'tr': 0.},
        PARTS: {'id': 0., 'tr': 1.},
        SUPERCON: {'sc': 1.}
    }
    def __init__(self,
                 losses_weights,  # {GLOBAL:{id, tr}, FOREGROUND:{id,tr}, ...}
                 triplet_margin=0.3,
                 loss_name='part_averaged_triplet_loss',
                 use_gpu=False,
                 num_classes=-1,
                 writer=None,
                 use_visibility_scores=False,
                 supercon_enabled=False,
                 supercon_mode="global",
                 supercon_weight=1.0,
                 supercon_weight_global=1.0,
                 supercon_weight_concat=1.0,
                 supercon_temperature=0.07,
                supercon_norm_type=2.0,
                supercon_hard_mining=False,
                supercon_hard_mining_ratio=0.5):
        super().__init__()
        self.use_gpu = use_gpu
        self.losses_weights = losses_weights
        self.part_triplet_loss = init_part_based_triplet_loss(loss_name, margin=triplet_margin, writer=writer)
        self.identity_loss = CrossEntropyLoss(label_smooth=True)
        self.supcon_loss = SupervisedContrastiveLoss(
                            temperature=supercon_temperature,
                            norm_type=supercon_norm_type,
                            hard_mining=supercon_hard_mining,
                            hard_mining_ratio=supercon_hard_mining_ratio
                        )
        self.supercon_enabled = supercon_enabled
        self.supercon_weight = supercon_weight
        self.supercon_weight_global = supercon_weight_global
        self.supercon_weight_concat = supercon_weight_concat
        self.supercon_mode = supercon_mode

        self.use_visibility_scores = use_visibility_scores

        # accuracy
        self.pred_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        if self.use_gpu:
            self.pred_accuracy = self.pred_accuracy.cuda()

    def forward(self, embeddings_dict, visibility_dict, id_cls_scores_dict, pids):
        """
        embeddings_dict: { 'globl':(N,D), 'parts':(N,K,D), ... }
        visibility_dict: { 'globl':(N), 'parts':(N,K), ... }
        id_cls_scores_dict: { 'globl':(N,num_classes), 'parts':(N,K,num_classes) ... }
        pids: (N,) 
        """
        losses = []
        summary = {}

        # 1) CE loss
        for key in [GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS]:
            w_ce = self.losses_weights[key].get('id', 0.0)
            if w_ce > 0:
                ce_loss, acc = self.compute_id_cls_loss(
                    id_cls_scores_dict[key], 
                    visibility_dict[key],
                    pids
                )
                losses.append((w_ce, ce_loss))
                info = summary.setdefault(key, OrderedDict())
                info['ce'] = ce_loss.item()
                info['acc'] = acc

        # 2) Triplet
        for key in [GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS]:
            w_tr = self.losses_weights[key].get('tr', 0.0)
            if w_tr > 0:
                tri_loss, triv_ratio, valid_ratio = self.compute_triplet_loss(
                    embeddings_dict[key], 
                    visibility_dict[key],
                    pids
                )
                losses.append((w_tr, tri_loss))
                info = summary.setdefault(key, OrderedDict())
                info['triplet'] = tri_loss.item()
                info['trivial'] = triv_ratio
                info['valid'] = valid_ratio

        # 3) SuperCon (옵션)
        if self.supercon_enabled and self.supercon_weight > 0:
            sc_val = self.compute_supercon_loss(embeddings_dict, pids)
            losses.append((self.supercon_weight, sc_val))
            summary.setdefault('supercon', {})['val'] = sc_val.item()

        # 최종 합산
        if len(losses) == 0:
            total_loss = torch.tensor(0., device=pids.device)
        else:
            total_loss = torch.stack([w*l for w, l in losses]).sum()

        return total_loss, summary

    # --------------------------------------------------
    # 아래 개별 함수들
    # --------------------------------------------------
    def compute_id_cls_loss(self, id_cls_scores, visibility_scores, pids):
        # scores: (N,num_classes) or (N,K,num_classes)
        if len(id_cls_scores.shape) == 3:
            # part-based
            N, K, C = id_cls_scores.shape
            id_cls_scores = id_cls_scores.flatten(0, 1)  # (N*K, C)
            pids = pids.unsqueeze(1).expand(-1, K).flatten(0, 1)  # (N*K)
            visibility = visibility_scores.flatten(0, 1) if visibility_scores.dim() == 2 else None
        else:
            visibility = None
            
        # visibility_scores를 weighting할 수도 있고, bool mask로 사용 가능
        weights = None
        if self.use_visibility_scores and visibility is not None:
            if visibility.dtype is torch.bool:
                id_cls_scores = id_cls_scores[visibility]
                pids = pids[visibility]
            elif visibility.dtype is not torch.bool:
                weights = visibility
                
        ce_loss = self.identity_loss(id_cls_scores, pids, weights)
        
        # accuracy
        preds = torch.argmax(id_cls_scores, dim=1)
        acc = (preds == pids).float().mean().item()
        return ce_loss, acc

    def compute_triplet_loss(self, embeddings, visibility, pids):
        # GiLt_loss.py의 안정적인 차원 처리 로직 통합
        # embeddings: (N,D) or (N,K,D)
        embeddings = embeddings if len(embeddings.shape) == 3 else embeddings.unsqueeze(1)
        
        if self.use_visibility_scores and visibility is not None:
            parts_visibility = visibility if len(visibility.shape) == 2 else visibility.unsqueeze(1)
        else:
            parts_visibility = None
            
        tri_loss, triv_ratio, valid_ratio = self.part_triplet_loss(
            embeddings, pids, parts_visibility=parts_visibility
        )
        return tri_loss, triv_ratio, valid_ratio

    def compute_supercon_loss(self, embeddings_dict, pids):
        if self.supercon_mode == "global":
            features = embeddings_dict[GLOBAL]
            return self.supcon_loss(features, pids) * self.supercon_weight_global
        elif self.supercon_mode == "concat_parts":
            features = embeddings_dict[CONCAT_PARTS]
            return self.supcon_loss(features, pids) * self.supercon_weight_concat
        elif self.supercon_mode == "both":
            features_global = embeddings_dict[GLOBAL]
            features_concat = embeddings_dict[CONCAT_PARTS]
            sc_loss_global = self.supcon_loss(features_global, pids) * self.supercon_weight_global
            sc_loss_concat = self.supcon_loss(features_concat, pids) * self.supercon_weight_concat
            return sc_loss_global + sc_loss_concat
        else:
            raise ValueError(f"Unknown supercon_mode: {self.supercon_mode}")
