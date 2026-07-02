"""
URL-BERT: Transformer-based Architecture for URL Analysis
Advanced transformer model specifically designed for URL understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer
from typing import Dict, List, Optional, Tuple
import math


class URLTokenizer:
    """Custom tokenizer for URL components"""
    
    def __init__(self, max_length: int = 200):
        self.max_length = max_length
        self.special_tokens = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
            '[PROTOCOL]': 4, '[DOMAIN]': 5, '[PATH]': 6, '[QUERY]': 7
        }
        
        # Build character vocabulary
        self.char_to_id = self.special_tokens.copy()
        
        # Add ASCII characters
        for i in range(32, 127):  # Printable ASCII
            char = chr(i)
            if char not in self.char_to_id:
                self.char_to_id[char] = len(self.char_to_id)
        
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
    
    def tokenize_url(self, url: str) -> Dict[str, torch.Tensor]:
        """Tokenize URL with component-aware segmentation"""
        # Parse URL components
        components = self._parse_url_components(url)
        
        # Convert to token IDs
        tokens = [self.char_to_id['[CLS]']]
        segment_ids = [0]
        
        # Add protocol
        if components['protocol']:
            tokens.append(self.char_to_id['[PROTOCOL]'])
            segment_ids.append(1)
            for char in components['protocol']:
                tokens.append(self.char_to_id.get(char, self.char_to_id['[UNK]']))
                segment_ids.append(1)
        
        # Add domain
        if components['domain']:
            tokens.append(self.char_to_id['[DOMAIN]'])
            segment_ids.append(2)
            for char in components['domain']:
                tokens.append(self.char_to_id.get(char, self.char_to_id['[UNK]']))
                segment_ids.append(2)
        
        # Add path
        if components['path']:
            tokens.append(self.char_to_id['[PATH]'])
            segment_ids.append(3)
            for char in components['path']:
                tokens.append(self.char_to_id.get(char, self.char_to_id['[UNK]']))
                segment_ids.append(3)
        
        # Add query
        if components['query']:
            tokens.append(self.char_to_id['[QUERY]'])
            segment_ids.append(4)
            for char in components['query']:
                tokens.append(self.char_to_id.get(char, self.char_to_id['[UNK]']))
                segment_ids.append(4)
        
        # Truncate if necessary
        if len(tokens) > self.max_length - 1:
            tokens = tokens[:self.max_length - 1]
            segment_ids = segment_ids[:self.max_length - 1]
        
        # Add SEP token
        tokens.append(self.char_to_id['[SEP]'])
        segment_ids.append(0)
        
        # Pad to max length
        attention_mask = [1] * len(tokens)
        while len(tokens) < self.max_length:
            tokens.append(self.char_to_id['[PAD]'])
            segment_ids.append(0)
            attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(segment_ids, dtype=torch.long)
        }
    
    def _parse_url_components(self, url: str) -> Dict[str, str]:
        """Parse URL into components"""
        components = {'protocol': '', 'domain': '', 'path': '', 'query': ''}
        
        # Simple URL parsing
        if '://' in url:
            protocol, rest = url.split('://', 1)
            components['protocol'] = protocol
        else:
            rest = url
        
        if '?' in rest:
            rest, query = rest.split('?', 1)
            components['query'] = query
        
        if '/' in rest:
            domain, path = rest.split('/', 1)
            components['domain'] = domain
            components['path'] = '/' + path
        else:
            components['domain'] = rest
        
        return components


class URLTransformerEncoder(nn.Module):
    """Custom Transformer Encoder for URL Analysis"""
    
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 num_hidden_layers: int = 6,
                 intermediate_size: int = 3072,
                 max_position_embeddings: int = 200,
                 dropout: float = 0.1):
        super().__init__()
        
        # Embeddings
        self.embeddings = URLEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )
        
        # Transformer layers
        self.encoder_layers = nn.ModuleList([
            URLTransformerLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout=dropout
            ) for _ in range(num_hidden_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> torch.Tensor:
        
        # Embeddings
        hidden_states = self.embeddings(input_ids, token_type_ids)
        
        # Extended attention mask
        extended_attention_mask = self._get_extended_attention_mask(attention_mask)
        
        # Transformer layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, extended_attention_mask)
        
        return self.layer_norm(hidden_states)
    
    def _get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Convert attention mask to extended format"""
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


class URLEmbeddings(nn.Module):
    """URL-specific embeddings with component awareness"""
    
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int, dropout: float):
        super().__init__()
        
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(5, hidden_size)  # 5 URL components
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class URLTransformerLayer(nn.Module):
    """Single transformer layer with URL-specific modifications"""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int, dropout: float):
        super().__init__()
        
        self.attention = URLMultiHeadAttention(hidden_size, num_attention_heads, dropout)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.layer_norm1(hidden_states + attention_output)
        
        # Feed-forward
        intermediate_output = F.gelu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.layer_norm2(attention_output + self.dropout(layer_output))
        
        return layer_output


class URLMultiHeadAttention(nn.Module):
    """Multi-head attention with URL component awareness"""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float):
        super().__init__()
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        
        # Attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return self.dense(context_layer)


class URLTransformer(nn.Module):
    """Complete URL Transformer model for phishing detection"""
    
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 num_hidden_layers: int = 6,
                 intermediate_size: int = 3072,
                 max_position_embeddings: int = 200,
                 num_classes: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.encoder = URLTransformerEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )
        
        # Classification head
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # Encoder
        sequence_output = self.encoder(input_ids, attention_mask, token_type_ids)
        
        # Pooling (use [CLS] token)
        pooled_output = torch.tanh(self.pooler(sequence_output[:, 0]))
        
        # Classification
        logits = self.classifier(pooled_output)
        probabilities = F.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'hidden_states': sequence_output,
            'pooled_output': pooled_output
        }