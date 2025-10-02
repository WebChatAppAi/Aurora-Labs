"""
Melody transformer model implementation.
Based on the original MelodyTransformer but with the new architecture.
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import load_file

from .base import BaseModel, BaseConfig, BaseTokenizer, ModelFactory, GenerationParams, GenerationResult


@dataclass
class MelodyConfig(BaseConfig):
    """Configuration for Melody Transformer model."""
    vocab_size: int = 901
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 12
    d_ff: int = 4096
    max_seq_len: int = 200
    dropout: float = 0.1

    # Special tokens
    bos_token: int = 0
    eos_token: int = 1
    pad_token: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MelodyConfig':
        """Create configuration from dictionary."""
        # Filter out fields that are not part of MelodyConfig
        valid_fields = {
            'vocab_size', 'd_model', 'n_heads', 'n_layers', 'd_ff', 
            'max_seq_len', 'dropout', 'bos_token', 'eos_token', 'pad_token'
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class MelodyTransformer(BaseModel):
    """Transformer model for melody generation."""

    def __init__(self, config: MelodyConfig):
        super().__init__(config)
        self.config = config

        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers
        )

        # Output projection
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        
        # Cache parameter count
        self._parameter_count = None

    def _init_weights(self, module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None):
        """Forward pass."""
        # Embeddings
        x = self.token_embedding(input_ids) * np.sqrt(self.config.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Create causal mask
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
            diagonal=1
        )

        # Apply attention mask if provided (for padding)
        if attention_mask is not None:
            # Create src_key_padding_mask for transformer
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Transformer forward pass
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)

        # Output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(
        self,
        start_tokens: List[int],
        params: GenerationParams,
        device: str = 'cpu'
    ) -> GenerationResult:
        """Generate melody tokens with advanced sampling."""
        start_time = time.time()
        self.eval()

        with torch.no_grad():
            generated = start_tokens.copy()
            input_ids = torch.tensor([generated], device=device)

            for step in range(params.max_length - len(start_tokens)):
                # Get predictions
                logits = self(input_ids)
                next_token_logits = logits[0, -1].clone()

                # Apply repetition penalty
                if params.repetition_penalty != 1.0 and len(generated) > 1:
                    for prev_token in set(generated):
                        if prev_token < len(next_token_logits):
                            if next_token_logits[prev_token] < 0:
                                next_token_logits[prev_token] *= params.repetition_penalty
                            else:
                                next_token_logits[prev_token] /= params.repetition_penalty

                # Apply temperature
                next_token_logits = next_token_logits / params.temperature

                # Apply top-k filtering
                if params.top_k > 0:
                    values, indices = torch.topk(next_token_logits, min(params.top_k, next_token_logits.size(-1)))
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, indices, values)

                # Apply nucleus (top-p) sampling
                if params.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > params.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                # Add to sequence
                generated.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

                # Stop on EOS token
                if next_token == self.config.eos_token:
                    break

        generation_time = time.time() - start_time

        return GenerationResult(
            tokens=generated,
            metadata={
                'generation_params': asdict(params),
                'generation_time': generation_time,
                'token_count': len(generated),
                'stopped_early': generated[-1] == self.config.eos_token if generated else False
            },
            generation_time=generation_time,
            model_info=self.get_model_info()
        )

    @property
    def model_type(self) -> str:
        """Get model type identifier."""
        return "MelodyTransformer"
    
    @property
    def parameter_count(self) -> int:
        """Get total number of parameters."""
        if self._parameter_count is None:
            self._parameter_count = sum(p.numel() for p in self.parameters())
        return self._parameter_count


class MelodyTokenizer(BaseTokenizer):
    """Melody tokenizer interface."""

    def __init__(self):
        # Import the actual tokenizer
        try:
            from alv_tokenizer import MelodyTokenizer as ActualTokenizer
            self._tokenizer = ActualTokenizer()
        except ImportError:
            raise ImportError("alv_tokenizer package is required for MelodyTokenizer")

    def tokenize(self, input_data) -> List[int]:
        """Convert input to tokens."""
        return self._tokenizer.tokenize(input_data)

    def detokenize(self, tokens: List[int]) -> bytes:
        """Convert tokens back to data."""
        return self._tokenizer.detokenize(tokens)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._tokenizer.vocab_size


class MelodyModelFactory(ModelFactory):
    """Factory for creating melody models."""

    @classmethod
    def create_model(cls, config: MelodyConfig) -> MelodyTransformer:
        """Create a melody model instance."""
        return MelodyTransformer(config)

    @classmethod
    def create_config(cls, config_data: Dict[str, Any]) -> MelodyConfig:
        """Create a melody config instance."""
        return MelodyConfig.from_dict(config_data)

    @classmethod
    def create_tokenizer(cls, tokenizer_config: Dict[str, Any]) -> MelodyTokenizer:
        """Create a melody tokenizer instance."""
        return MelodyTokenizer()

    @classmethod
    def supported_model_types(cls) -> List[str]:
        """Get list of supported model types."""
        return ["MelodyTransformer"]

    @classmethod
    def load_model_from_files(
        cls,
        model_path: str,
        config_path: str,
        device: str = 'cpu'
    ) -> MelodyTransformer:
        """Load model from safetensors and config files."""
        # Load config
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        config = cls.create_config(config_data)

        # Create model
        model = cls.create_model(config)

        # Load weights
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)

        # Move to device
        model = model.to(device)
        model.eval()

        return model