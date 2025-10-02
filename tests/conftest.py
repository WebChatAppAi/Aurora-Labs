"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock

# Test configuration data
TEST_CONFIG_DATA = {
    'api': {
        'host': '0.0.0.0',
        'port': 8000,
        'title': 'Test API',
        'version': '1.0.0',
        'description': 'Test API'
    },
    'storage': {
        'models_root': './test_models',
        'cache_enabled': True,
        'max_loaded_models': 2,
        'auto_unload_after': 3600
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
            'tags': ['test'],
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
def test_config_data():
    """Test configuration data."""
    return TEST_CONFIG_DATA.copy()


@pytest.fixture
def temp_config_file():
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(TEST_CONFIG_DATA, f)
        yield f.name

    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.vocab_size = 901
    tokenizer.tokenize.return_value = [0, 100, 200, 300, 1]
    tokenizer.detokenize.return_value = b'fake_midi_data'

    # Mock vocabulary
    vocab = Mock()
    vocab.decode_token.return_value = {
        'type': 'melody_note',
        'pitch': 60,
        'duration': 4,
        'velocity': 'MEDIUM'
    }
    tokenizer.vocabulary = vocab

    return tokenizer


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.model_type = 'MelodyTransformer'
    model.parameter_count = 1000000

    # Mock config
    config = Mock()
    config.vocab_size = 901
    config.bos_token = 0
    config.eos_token = 1
    config.pad_token = 2
    model.config = config

    # Mock generation
    generation_result = Mock()
    generation_result.tokens = [0, 100, 200, 300, 1]
    generation_result.generation_time = 0.5
    generation_result.metadata = {'stopped_early': False}
    model.generate.return_value = generation_result

    model.get_model_info.return_value = {
        'type': 'MelodyTransformer',
        'parameters': 1000000,
        'config': {'vocab_size': 901}
    }

    return model


@pytest.fixture
def mock_loaded_model(mock_model, mock_tokenizer):
    """Mock loaded model container."""
    from app.services.model_registry import LoadedModel
    from datetime import datetime

    loaded_model = Mock()
    loaded_model.model = mock_model
    loaded_model.tokenizer = mock_tokenizer
    loaded_model.load_time = datetime.now()
    loaded_model.last_accessed = datetime.now()
    loaded_model.access_count = 0

    # Mock config
    config = Mock()
    config.architecture = Mock()
    config.architecture.bos_token = 0
    loaded_model.config = config

    return loaded_model


@pytest.fixture
def sample_midi_bytes():
    """Sample MIDI file bytes for testing."""
    # This is a minimal MIDI file header
    return b'MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\x60MTrk\x00\x00\x00\x0b\x00\x90\x3c\x40\x48\x80\x3c\x40\x00\xff\x2f\x00'


@pytest.fixture
def sample_notes():
    """Sample notes data for testing."""
    return [
        {
            'pitch': 60,
            'start_time': 0.0,
            'duration': 1.0,
            'velocity': 80
        },
        {
            'pitch': 62,
            'start_time': 1.0,
            'duration': 1.0,
            'velocity': 90
        },
        {
            'pitch': 64,
            'start_time': 2.0,
            'duration': 1.0,
            'velocity': 85
        }
    ]


@pytest.fixture
def sample_tokens():
    """Sample token sequence for testing."""
    return [0, 100, 150, 200, 250, 300, 1]  # BOS, notes..., EOS


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark integration tests
        if "test_api" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

        # Mark slow tests
        if "test_generate" in item.name or "test_load_model" in item.name:
            item.add_marker(pytest.mark.slow)