from __future__ import division, print_function, absolute_import

import os.path as osp
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.cuda import amp

from ...losses.GiLt_con_loss import GiLtconLoss
from ...losses.body_part_attention_loss import BodyPartAttentionLoss
from ...metrics.distance import compute_distance_matrix_using_bp_features
from ...utils import (
    plot_body_parts_pairs_distance_distribution,
    plot_pairs_distance_distribution,
    re_ranking,
)
from ...utils.tools import extract_test_embeddings
from ...utils.torchtools import collate
from ...utils.visualization.feature_map_visualization import display_feature_maps
from ...utils.constants import (
    GLOBAL,
    PARTS,
    PIXELS,
    FOREGROUND,
    CONCAT_PARTS,
)

from ..engine import Engine
from ... import metrics


class ImageSupConEngine2(Engine):
    r"""Training/testing engine for part-based image-reid."""

    def __init__(
        self,
        config,
        datamanager,
        model,
        optimizer,
        writer,
        loss_name,
        dist_combine_strat,
        batch_size_pairwise_dist_matrix,
        engine_state,
        margin=0.3,
        scheduler=None,
        use_gpu=True,
        save_model_flag=False,
        mask_filtering_training=False,
        mask_filtering_testing=False,
        accumulation_steps=1,
    ):
        super(ImageSupConEngine2, self).__init__(
            config,
            datamanager,
            writer,
            engine_state,
            use_gpu=use_gpu,
            save_model_flag=save_model_flag,
            detailed_ranking=config.test.detailed_ranking,
        )

        self.model = model
        self.register_model("model", model, optimizer, scheduler)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parts_num = self.config.model.kpr.masks.parts_num
        self.mask_filtering_training = mask_filtering_training
        self.mask_filtering_testing = mask_filtering_testing
        self.dist_combine_strat = dist_combine_strat
        self.batch_size_pairwise_dist_matrix = batch_size_pairwise_dist_matrix
        self.mixed_precision = self.config.train.mixed_precision
        self.scaler = amp.GradScaler() if self.mixed_precision else None
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

        self.losses_weights = self.config.loss.part_based.weights
        self.GiLt = GiLtconLoss(
            losses_weights=self.losses_weights,
            use_visibility_scores=self.mask_filtering_training,
            triplet_margin=margin,
            loss_name=loss_name,
            writer=self.writer,
            use_gpu=self.use_gpu,
            num_classes=datamanager.num_train_pids,
            supercon_enabled=self.config.loss.supercon.enabled,
            supercon_weight=self.config.loss.supercon.weight,
            supercon_temperature=self.config.loss.supercon.temperature,
        )

        self.body_part_attention_loss = BodyPartAttentionLoss(
            loss_type=self.config.loss.part_based.ppl,
            use_gpu=self.use_gpu,
            best_pred_ratio=self.config.loss.part_based.best_pred_ratio,
            num_classes=self.parts_num + 1,
        )
        self.feature_extraction_timer = self.writer.feature_extraction_timer
        self.loss_timer = self.writer.loss_timer
        self.optimizer_timer = self.writer.optimizer_timer

    def forward_backward(self, data):
        imgs, target_masks, prompt_masks, keypoints_xyc, pids, imgs_path, cam_id = self.parse_data_for_train(data)
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        with amp.autocast(enabled=self.mixed_precision):
            (
                embeddings_dict,
                visibility_scores_dict,
                id_cls_scores_dict,
                pixels_cls_scores,
                spatial_features,
                masks,
            ) = self.model(
                imgs,
                target_masks=target_masks,
                prompt_masks=prompt_masks,
                keypoints_xyc=keypoints_xyc,
                cam_label=cam_id
            )
            display_feature_maps(
                embeddings_dict, spatial_features, masks[PARTS], imgs_path, pids
            )

            loss, loss_summary = self.combine_losses(
                visibility_scores_dict,
                embeddings_dict,
                id_cls_scores_dict,
                pids,
                pixels_cls_scores,
                target_masks,
                bpa_weight=self.losses_weights[PIXELS]["ce"],
            )

        self.optimizer_timer.start()
        if self.scaler is None:
            loss.backward()
            self.step_count += 1
            if self.step_count % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            self.scaler.scale(loss).backward()
            self.step_count += 1
            if self.step_count % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        self.optimizer_timer.stop()

        return loss, loss_summary

    def combine_losses(
        self,
        visibility_scores_dict,
        embeddings_dict,
        id_cls_scores_dict,
        pids,
        pixels_cls_scores=None,
        target_masks=None,
        bpa_weight=0,
    ):
        loss, loss_summary = self.GiLt(
            embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pids
        )

        if (
            pixels_cls_scores is not None
            and target_masks is not None
            and bpa_weight > 0
        ):
            pixels_cls_score_targets = target_masks.argmax(dim=1)  
            bpa_loss, bpa_loss_summary = self.body_part_attention_loss(
                pixels_cls_scores, pixels_cls_score_targets
            )
            loss += bpa_weight * bpa_loss
            loss_summary = {**loss_summary, **bpa_loss_summary}

        return loss, loss_summary

    # Other methods remain the same as in supcontrastive_engine.py
