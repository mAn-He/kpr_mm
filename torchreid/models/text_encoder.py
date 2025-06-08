import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class TextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", frozen=True, embed_dim=512):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        
        # 원본 출력 차원
        self.original_dim = self.model.config.hidden_size
        
        # 필요하다면 차원 조정을 위한 프로젝션 레이어
        self.projection = None
        if embed_dim != self.original_dim:
            self.projection = nn.Linear(self.original_dim, embed_dim)
        
        # 인코더 고정 (추론 속도 향상 및 과적합 방지)
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, descriptions):
        if descriptions is None or len(descriptions) == 0:
            # 설명이 없으면 적절한 크기의 영벡터 반환
            device = next(self.model.parameters()).device
            if self.projection is not None:
                return torch.zeros(1, self.projection.out_features, device=device)
            else:
                return torch.zeros(1, self.original_dim, device=device)
        
        # 텍스트 인코딩
        inputs = self.tokenizer(descriptions, padding=True, truncation=True, 
                              return_tensors="pt").to(next(self.model.parameters()).device)
        
        outputs = self.model(**inputs)
        
        # [CLS] 토큰 임베딩 사용
        text_embeddings = outputs.last_hidden_state[:, 0]
        
        # 필요시 차원 조정
        if self.projection is not None:
            text_embeddings = self.projection(text_embeddings)
        
        return text_embeddings