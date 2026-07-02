"""
URLNet: CNN-based Architecture for Phishing URL Detection
Advanced character-level analysis with attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


class CharacterEmbedding(nn.Module):
    """Advanced character embedding with positional encoding"""
    
    def __init__(self, vocab_size: int = 128, embed_dim: int = 128, max_length: int = 200):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_length, embed_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def _create_positional_encoding(self, max_length: int, embed_dim: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Character embedding
        embedded = self.char_embedding(x) * np.sqrt(self.embed_dim)
        
        # Add positional encoding
        if seq_len <= self.max_length:
            pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
            embedded = embedded + pos_enc
        
        return self.dropout(embedded)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for URL components"""
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        return self.out_proj(context)


class URLNet(nn.Module):
    """
    Advanced URLNet Architecture for Phishing Detection
    Combines CNN, attention mechanisms, and feature fusion
    """
    
    def __init__(self, 
                 vocab_size: int = 128,
                 embed_dim: int = 128,
                 num_filters: int = 256,
                 filter_sizes: List[int] = [3, 4, 5, 6],
                 num_heads: int = 8,
                 hidden_dim: int = 512,
                 num_classes: int = 2,
                 max_length: int = 200,
                 dropout: float = 0.3):
        super().__init__()
        
        self.max_length = max_length
        
        # Character embedding with positional encoding
        self.embedding = CharacterEmbedding(vocab_size, embed_dim, max_length)
        
        # Multi-scale CNN layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs, padding=fs//2)
            for fs in filter_sizes
        ])
        
        # Batch normalization for each conv layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in filter_sizes
        ])
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Feature fusion layers
        total_filters = len(filter_sizes) * num_filters
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_filters + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        
        # Character embedding with positional encoding
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Multi-head attention
        attended = self.attention(embedded)  # [batch_size, seq_len, embed_dim]
        
        # Global average pooling for attention features
        if lengths is not None:
            # Mask padding tokens
            mask = torch.arange(attended.size(1)).expand(batch_size, -1).to(x.device)
            mask = mask < lengths.unsqueeze(1)
            attended_masked = attended * mask.unsqueeze(-1).float()
            attention_features = attended_masked.sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            attention_features = attended.mean(dim=1)  # [batch_size, embed_dim]
        
        # CNN features
        embedded_transposed = embedded.transpose(1, 2)  # [batch_size, embed_dim, seq_len]
        
        conv_features = []
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            # Convolution + BatchNorm + ReLU + MaxPool
            conv_out = F.relu(bn(conv(embedded_transposed)))
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            conv_features.append(pooled.squeeze(-1))
        
        # Concatenate all CNN features
        cnn_features = torch.cat(conv_features, dim=1)  # [batch_size, total_filters]
        
        # Feature fusion
        combined_features = torch.cat([cnn_features, attention_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        probabilities = F.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'features': fused_features,
            'attention_weights': None  # Can be extracted from attention layer if needed
        }


class URLNetLoss(nn.Module):
    """Custom loss function with focal loss for imbalanced data"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross entropy with label smoothing
        ce_loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing, reduction='none')
        
        # Focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


def create_urlnet_model(config: Dict) -> URLNet:
    """Factory function to create URLNet model with configuration"""
    return URLNet(
        vocab_size=config.get('vocab_size', 128),
        embed_dim=config.get('embed_dim', 128),
        num_filters=config.get('num_filters', 256),
        filter_sizes=config.get('filter_sizes', [3, 4, 5, 6]),
        num_heads=config.get('num_heads', 8),
        hidden_dim=config.get('hidden_dim', 512),
        num_classes=config.get('num_classes', 2),
        max_length=config.get('max_length', 200),
        dropout=config.get('dropout', 0.3)
    )