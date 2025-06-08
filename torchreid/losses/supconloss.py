from __future__ import division, print_function, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchreid import metrics
from torchreid.losses import CrossEntropyLoss

class SupervisedContrastiveLoss(nn.Module):
    r"""
    Supervised Contrastive Loss
    (https://arxiv.org/abs/2004.11362)

    기본 아이디어:
      - feature를 L2 normalize
      - similarity matrix 계산 후 temperature로 스케일링
      - 같은 라벨이면 positive pair, 다른 라벨이면 negative pair
      - 자기 자신(대각선) 제외
      - positive인 항목만 골라 평균 negative log-likelihood 계산
      
    확장 기능:
      - 다양한 정규화 전략 지원 (norm_type 파라미터)
      - 하드 네거티브 마이닝 지원 (hard_mining 옵션)
    """
    def __init__(self, temperature=0.07, norm_type=2.0, 
                 hard_mining=False, hard_mining_ratio=0.5):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.norm_type = norm_type  # 정규화 타입 (1.0=L1, 2.0=L2, 등)
        self.hard_mining = hard_mining  # 하드 마이닝 사용 여부
        self.hard_mining_ratio = hard_mining_ratio  # 상위 몇 %를 하드로 간주할지

    def forward(self, features, labels=None, mask=None):
        """
        features: (N, D)
        labels:   (N,)  (ID 레이블)
        mask:     (N,N) optional 커스텀 마스크
        """
        device = features.device
        # 1) 정규화 (norm_type에 따라 달라짐)
        features = F.normalize(features, p=self.norm_type, dim=1)
        N = features.size(0)

        # 2) sim_matrix: (N,N)
        sim_matrix = torch.matmul(features, features.t())
        sim_matrix = sim_matrix / self.temperature

        # 3) 라벨 기반 마스크
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.size(0) != N:
                raise ValueError("Num of labels != num of features")
            mask = torch.eq(labels, labels.t()).float().to(device)
        elif mask is not None:
            mask = mask.float().to(device)
        else:
            mask = torch.ones_like(sim_matrix).to(device)

        # 4) 자기 자신 제외 (대각선 제거)
        diag = torch.eye(N, device=device)
        mask_pos = mask - diag  # positive pair 마스크
        mask_neg = 1.0 - mask   # negative pair 마스크
        
        # 5) 하드 마이닝 적용 (활성화된 경우)
        if self.hard_mining:
            # 각 샘플마다 네거티브 유사도 계산
            neg_sim = sim_matrix * mask_neg
            
            # 하드 네거티브 마스크 생성
            hard_neg_weights = torch.zeros_like(neg_sim)
            for i in range(N):
                # i번째 행의 네거티브 유사도 값들
                neg_sims_i = neg_sim[i]
                # 0보다 큰 값들만 선택 (실제 네거티브)
                valid_negs = neg_sims_i[mask_neg[i] > 0]
                if len(valid_negs) > 0:
                    # 상위 hard_mining_ratio만큼의 임계값 계산
                    k = max(1, int(len(valid_negs) * self.hard_mining_ratio))
                    topk_values, _ = torch.topk(valid_negs, k)
                    threshold = topk_values[-1]
                    # 임계값 이상인 값들만 1로 설정
                    hard_neg_weights[i] = (neg_sims_i >= threshold).float() * mask_neg[i]
            
            # 원래 마스크 대신 하드 네거티브 마스크 사용
            neg_mask_for_exp = hard_neg_weights
        else:
            neg_mask_for_exp = mask_neg

        # 6) log-softmax 계산
        exp_sim = torch.exp(sim_matrix)
        # 기존 방식: exp_sim.sum(dim=1, keepdim=True)
        # 수정: 네거티브 샘플만 합산 (하드 마이닝 적용 시 하드 네거티브만)
        neg_exp_sum = torch.sum(exp_sim * neg_mask_for_exp, dim=1, keepdim=True)
        # 자기 자신도 분모에 포함
        self_exp = torch.sum(diag * exp_sim, dim=1, keepdim=True)
        denominator = neg_exp_sum + self_exp + 1e-10
        
        log_prob = sim_matrix - torch.log(denominator)

        # 7) positive pair에 대해서만 평균
        pos_count = mask_pos.sum(dim=1) + 1e-10
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / pos_count

        loss = -mean_log_prob_pos.mean()
        return loss