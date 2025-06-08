import torch
import torch.nn as nn
import torch.nn.functional as F
from torchreid.models.compact_bilinear_pooling import CompactBilinearPooling

class FusionFactory:
    @staticmethod
    def create_fusion_module(fusion_type, image_dim, text_dim, output_dim, alpha=0.7):
        if fusion_type == "concat":
            return ConcatFusion(image_dim, text_dim, output_dim)
        elif fusion_type == "bilinear":
            return BilinearFusion(image_dim, text_dim, output_dim)
        elif fusion_type == "attention":
            return CrossAttentionFusion(image_dim, text_dim, output_dim)
        elif fusion_type == "late":
            return LateFusion(image_dim, text_dim, alpha)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

class ConcatFusion(nn.Module):
    def __init__(self, image_dim, text_dim, output_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + text_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, image_embed, text_embed):
        combined = torch.cat([image_embed, text_embed], dim=1)
        return self.fusion(combined)

class BilinearFusion(nn.Module):
    def __init__(self, image_dim, text_dim, output_dim):
        super().__init__()
        self.cbp = CompactBilinearPooling(image_dim, text_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, image_embed, text_embed):
        fused = self.cbp(image_embed, text_embed)
        return self.norm(fused)

class CrossAttentionFusion(nn.Module):
    def __init__(self, image_dim, text_dim, output_dim):
        super().__init__()
        # Adjust dimensions if needed
        self.image_proj = nn.Linear(image_dim, output_dim) if image_dim != output_dim else nn.Identity()
        self.text_proj = nn.Linear(text_dim, output_dim) if text_dim != output_dim else nn.Identity()
        
        # Cross attention components
        self.query_proj = nn.Linear(output_dim, output_dim)
        self.key_proj = nn.Linear(output_dim, output_dim)
        self.value_proj = nn.Linear(output_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, image_embed, text_embed):
        # Project to common dimension if needed
        image_embed = self.image_proj(image_embed)
        text_embed = self.text_proj(text_embed)
        
        # Reshape for attention
        batch_size = image_embed.size(0)
        image_embed_2d = image_embed.view(batch_size, -1, image_embed.size(-1))
        text_embed_2d = text_embed.unsqueeze(1) if text_embed.dim() == 2 else text_embed
        
        # Project queries, keys and values
        queries = self.query_proj(image_embed_2d)
        keys = self.key_proj(text_embed_2d)
        values = self.value_proj(text_embed_2d)
        
        # Calculate attention
        attn = torch.bmm(queries, keys.transpose(1, 2)) / (keys.size(-1) ** 0.5)
        attn = F.softmax(attn, dim=2)
        
        # Apply attention to values
        output = torch.bmm(attn, values)
        output = self.output_proj(output)
        output = output + image_embed_2d  # Residual connection
        output = self.norm(output)
        
        # Reshape back
        output = output.view_as(image_embed)
        return output

class LateFusion(nn.Module):
    def __init__(self, image_dim, text_dim, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        # Optional projections if dimensions don't match
        self.image_proj = None if image_dim == text_dim else nn.Linear(image_dim, text_dim)
        self.text_proj = None if image_dim == text_dim else nn.Linear(text_dim, image_dim)
        self.output_dim = image_dim if self.image_proj is None else text_dim
        
    def forward(self, image_embed, text_embed):
        # Normalize embeddings
        image_embed_norm = F.normalize(image_embed, p=2, dim=1)
        text_embed_norm = F.normalize(text_embed, p=2, dim=1)
        
        # Project if needed
        if self.image_proj is not None:
            image_embed_norm = self.image_proj(image_embed_norm)
        elif self.text_proj is not None:
            text_embed_norm = self.text_proj(text_embed_norm)
        
        # Weighted combination
        fused_embeddings = self.alpha * image_embed_norm + (1 - self.alpha) * text_embed_norm
        fused_embeddings = F.normalize(fused_embeddings, p=2, dim=1)
        return fused_embeddings