"""
Unit tests for configuration management.
"""

import pytest
import yaml
import tempfile
from pathlib import Path

from app.core.config import ConfigManager, AppConfig, ModelConfig, GenerationDefaults


class TestConfigManager:
    """Test cases for ConfigManager."""

    @pytest.fixture
    def sample_config_data(self):
        """Sample configuration data for testing."""
        return {
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'title': 'Test API',
                'version': '1.0.0',
                'description': 'Test description'
            },
            'storage': {
                'models_root': './test_models',
                'cache_enabled': True,
                'max_loaded_models': 2,
                'auto_unload_after': 1800
            },
            'tokenizers': {
                'test_tokenizer': {
                    'type': 'MelodyTokenizer',
                    'min_pitch': 48,
                    'max_pitch': 84,
                    'time_resolution': 16
                }
            },
            'models': {
                'test_model': {
                    'type': 'MelodyTransformer',
                    'tokenizer': 'test_tokenizer',
                    'model_file': 'test_model.safetensors',
                    'config_file': 'test_config.json',
                    'architecture': {
                        'vocab_size': 901,
                        'd_model': 512,
                        'n_heads': 8,
                        'n_layers': 6,
                        'd_ff': 2048,
                        'max_seq_len': 200,
                        'dropout': 0.1
                    },
                    'generation_defaults': {
                        'temperature': 1.0,
                        'top_k': 50,
                        'top_p': 0.9,
                        'repetition_penalty': 1.1,
                        'max_length': 120
                    },
                    'tags': ['test', 'small'],
                    'description': 'Test model'
                }
            },
            'generation_profiles': {
                'test_profile': {
                    'temperature': 1.2,
                    'top_k': 60,
                    'top_p': 0.95,
                    'repetition_penalty': 1.0,
                    'description': 'Test profile'
                }
            },
            'output_formats': {
                'test_format': {
                    'include_notes': True,
                    'include_midi': True,
                    'include_tokens': False,
                    'note_format': 'beats'
                }
            },
            'logging': {
                'level': 'INFO',
                'file': 'test.log',
                'max_size': '10MB',
                'backup_count': 3,
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            },
            'performance': {
                'batch_size_limit': 5,
                'sequence_length_limit': 500,
                'generation_timeout': 30,
                'memory_threshold': 0.8
            },
            'security': {
                'rate_limit': '50/minute',
                'max_file_size': '5MB',
                'allowed_file_types': ['.mid', '.midi'],
                'cors_origins': ['*']
            }
        }

    @pytest.fixture
    def temp_config_file(self, sample_config_data):
        """Create temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config_data, f)
            return f.name

    def test_load_config_success(self, temp_config_file):
        """Test successful configuration loading."""
        config_manager = ConfigManager(temp_config_file)
        config = config_manager.config

        assert isinstance(config, AppConfig)
        assert config.api.title == 'Test API'
        assert config.storage.max_loaded_models == 2
        assert len(config.models) == 1
        assert 'test_model' in config.models

        # Cleanup
        Path(temp_config_file).unlink()

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            ConfigManager('non_existent_config.yaml')

    def test_get_model_config(self, temp_config_file):
        """Test getting specific model configuration."""
        config_manager = ConfigManager(temp_config_file)

        model_config = config_manager.get_model_config('test_model')
        assert model_config is not None
        assert model_config.type == 'MelodyTransformer'
        assert model_config.architecture.vocab_size == 901

        # Test non-existent model
        assert config_manager.get_model_config('non_existent') is None

        # Cleanup
        Path(temp_config_file).unlink()

    def test_validate_model_files(self, temp_config_file):
        """Test model file validation."""
        config_manager = ConfigManager(temp_config_file)

        # Should return False since files don't exist
        assert not config_manager.validate_model_files('test_model')

        # Cleanup
        Path(temp_config_file).unlink()


class TestModelConfig:
    """Test cases for ModelConfig validation."""

    def test_valid_model_config(self):
        """Test creating valid model configuration."""
        config_data = {
            'type': 'MelodyTransformer',
            'tokenizer': 'test_tokenizer',
            'model_file': 'model.safetensors',
            'config_file': 'config.json',
            'architecture': {
                'vocab_size': 901,
                'd_model': 512,
                'n_heads': 8,
                'n_layers': 6,
                'd_ff': 2048,
                'max_seq_len': 200,
                'dropout': 0.1
            },
            'generation_defaults': {
                'temperature': 1.0,
                'top_k': 50,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'max_length': 120
            },
            'tags': ['test'],
            'description': 'Test model'
        }

        model_config = ModelConfig(**config_data)
        assert model_config.type == 'MelodyTransformer'
        assert model_config.architecture.vocab_size == 901

    def test_invalid_model_type(self):
        """Test invalid model type validation."""
        config_data = {
            'type': 'InvalidTransformer',  # Invalid type
            'tokenizer': 'test_tokenizer',
            'model_file': 'model.safetensors',
            'config_file': 'config.json',
            'architecture': {
                'vocab_size': 901,
                'd_model': 512,
                'n_heads': 8,
                'n_layers': 6,
                'd_ff': 2048,
                'max_seq_len': 200
            },
            'generation_defaults': {
                'temperature': 1.0,
                'top_k': 50,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'max_length': 120
            }
        }

        with pytest.raises(ValueError, match="Model type must be one of"):
            ModelConfig(**config_data)


class TestGenerationDefaults:
    """Test cases for GenerationDefaults validation."""

    def test_valid_generation_defaults(self):
        """Test creating valid generation defaults."""
        params = GenerationDefaults(
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            max_length=120
        )

        assert params.temperature == 1.0
        assert params.top_k == 50
        assert params.top_p == 0.9