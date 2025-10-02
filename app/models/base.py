"""
Abstract base classes for AI music models.
Provides a common interface for different model architectures.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class GenerationParams:
    """Parameters for music generation."""
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_length: int = 120
    seed_tokens: Optional[List[int]] = None


@dataclass
class GenerationResult:
    """Result of music generation."""
    tokens: List[int]
    metadata: Dict[str, Any]
    generation_time: float
    model_info: Dict[str, Any]


class BaseConfig(ABC):
    """Abstract base configuration class."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from dictionary."""
        pass


class BaseTokenizer(ABC):
    """Abstract base tokenizer class."""

    @abstractmethod
    def tokenize(self, input_data: Union[str, bytes]) -> List[int]:
        """Convert input to tokens."""
        pass

    @abstractmethod
    def detokenize(self, tokens: List[int]) -> bytes:
        """Convert tokens back to data."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        pass


class BaseModel(nn.Module, ABC):
    """Abstract base model class for music generation."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    @abstractmethod
    def generate(
        self,
        start_tokens: List[int],
        params: GenerationParams,
        device: str = 'cpu'
    ) -> GenerationResult:
        """Generate music tokens."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Get model type identifier."""
        pass

    @property
    def parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'type': self.model_type,
            'parameters': self.parameter_count,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        }


class ModelFactory(ABC):
    """Abstract factory for creating models."""

    @classmethod
    @abstractmethod
    def create_model(cls, config: BaseConfig) -> BaseModel:
        """Create a model instance."""
        pass

    @classmethod
    @abstractmethod
    def create_config(cls, config_data: Dict[str, Any]) -> BaseConfig:
        """Create a config instance."""
        pass

    @classmethod
    @abstractmethod
    def create_tokenizer(cls, tokenizer_config: Dict[str, Any]) -> BaseTokenizer:
        """Create a tokenizer instance."""
        pass

    @classmethod
    @abstractmethod
    def supported_model_types(cls) -> List[str]:
        """Get list of supported model types."""
        pass