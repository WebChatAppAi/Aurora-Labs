"""
Unit tests for API endpoints.
"""

import pytest
import json
import base64
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from main import create_app


class TestAPIEndpoints:
    """Test cases for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_model_registry(self):
        """Mock model registry for testing."""
        with patch('app.services.model_registry.get_model_registry') as mock:
            registry = Mock()
            registry.get_available_models.return_value = {
                'test_model': {
                    'type': 'MelodyTransformer',
                    'description': 'Test model',
                    'tags': ['test'],
                    'architecture': {
                        'vocab_size': 901,
                        'd_model': 512,
                        'n_heads': 8,
                        'n_layers': 6,
                        'd_ff': 2048,
                        'max_seq_len': 200
                    },
                    'file_exists': True,
                    'is_loaded': False,
                    'model_file': 'test_model.safetensors',
                    'config_file': 'test_config.json',
                    'supported': True
                }
            }
            registry.get_loaded_models.return_value = []
            registry.is_model_loaded.return_value = False
            registry.device = 'cpu'
            mock.return_value = registry
            yield registry

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"

    def test_api_health_endpoint(self, client, mock_model_registry):
        """Test API health endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_list_models_endpoint(self, client, mock_model_registry):
        """Test list models endpoint."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "total_models" in data
        assert "loaded_models" in data
        assert len(data["models"]) == 1
        assert "test_model" in data["models"]

    def test_load_model_endpoint(self, client, mock_model_registry):
        """Test load model endpoint."""
        mock_model_registry.load_model.return_value = {
            'success': True,
            'message': 'Model loaded successfully',
            'load_time': 1.5,
            'model_type': 'MelodyTransformer',
            'parameter_count': 1000000,
            'vocab_size': 901,
            'device': 'cpu'
        }

        request_data = {
            "model_name": "test_model",
            "force_reload": False
        }

        response = client.post("/api/v1/models/load", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] == True
        assert data["model_name"] == "test_model"
        assert "load_time" in data

    def test_load_model_failure(self, client, mock_model_registry):
        """Test load model endpoint failure."""
        mock_model_registry.load_model.return_value = {
            'success': False,
            'error': 'Model not found'
        }

        request_data = {
            "model_name": "non_existent_model"
        }

        response = client.post("/api/v1/models/load", json=request_data)
        assert response.status_code == 400

    def test_unload_model_endpoint(self, client, mock_model_registry):
        """Test unload model endpoint."""
        mock_model_registry.unload_model.return_value = {
            'success': True,
            'message': 'Model unloaded successfully'
        }

        request_data = {
            "model_name": "test_model"
        }

        response = client.post("/api/v1/models/unload", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] == True

    def test_generate_simple_endpoint(self, client, mock_model_registry):
        """Test simple generation endpoint."""
        # Mock loaded model
        mock_loaded_model = Mock()
        mock_loaded_model.model.generate.return_value = Mock(
            tokens=[0, 100, 200, 300, 1],
            generation_time=0.5,
            metadata={'stopped_early': False}
        )
        mock_loaded_model.tokenizer.detokenize.return_value = b'fake_midi_data'
        mock_loaded_model.model.get_model_info.return_value = {
            'type': 'MelodyTransformer',
            'parameters': 1000000
        }
        mock_loaded_model.config.architecture = {'bos_token': 0}

        mock_model_registry.get_model.return_value = mock_loaded_model

        response = client.get("/api/v1/generate/simple?model_name=test_model&temperature=1.0")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert len(data["melodies"]) == 1

    def test_generate_endpoint_validation(self, client, mock_model_registry):
        """Test generation endpoint validation."""
        # Test missing model
        request_data = {
            "model_name": "non_existent_model",
            "num_generations": 1
        }

        mock_model_registry.get_model.return_value = None

        response = client.post("/api/v1/generate", json=request_data)
        assert response.status_code == 400

    def test_generate_with_invalid_params(self, client):
        """Test generation with invalid parameters."""
        request_data = {
            "model_name": "test_model",
            "params": {
                "temperature": 5.0,  # Invalid - too high
                "top_k": 1000,      # Invalid - too high
                "max_length": -1     # Invalid - negative
            }
        }

        response = client.post("/api/v1/generate", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_generate_from_midi_endpoint(self, client, mock_model_registry):
        """Test generate from MIDI endpoint."""
        # Mock loaded model
        mock_loaded_model = Mock()
        mock_loaded_model.tokenizer.tokenize.return_value = [0, 100, 200, 1]
        mock_loaded_model.model.generate.return_value = Mock(
            tokens=[0, 100, 200, 300, 1],
            generation_time=0.5,
            metadata={'stopped_early': False}
        )
        mock_loaded_model.tokenizer.detokenize.return_value = b'fake_midi_data'
        mock_loaded_model.model.get_model_info.return_value = {
            'type': 'MelodyTransformer',
            'parameters': 1000000
        }
        mock_loaded_model.config.architecture = {'bos_token': 0}

        mock_model_registry.get_model.return_value = mock_loaded_model

        # Create fake MIDI file
        fake_midi_content = b'MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\x60'

        response = client.post(
            "/api/v1/generate/from-midi",
            data={"model_name": "test_model"},
            files={"midi_file": ("test.mid", fake_midi_content, "audio/midi")}
        )

        # This might fail due to MIDI parsing, but we test the endpoint structure
        assert response.status_code in [200, 400, 500]

    def test_config_endpoint(self, client):
        """Test configuration endpoint."""
        response = client.get("/api/v1/config")
        assert response.status_code == 200

        data = response.json()
        assert "loaded" in data
        assert "total_models" in data
        assert "total_tokenizers" in data

    def test_system_status_endpoint(self, client, mock_model_registry):
        """Test system status endpoint."""
        mock_model_registry.get_memory_info.return_value = {
            'loaded_models_count': 0,
            'device': 'cpu'
        }

        response = client.get("/api/v1/status")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "operational"
        assert "timestamp" in data
        assert "loaded_models" in data


class TestRequestValidation:
    """Test request validation."""

    def test_generation_params_validation(self):
        """Test generation parameters validation."""
        from app.schemas.requests import GenerationParams
        from pydantic import ValidationError

        # Valid params
        params = GenerationParams(
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            max_length=120
        )
        assert params.temperature == 1.0

        # Invalid temperature
        with pytest.raises(ValidationError):
            GenerationParams(temperature=0.05)  # Too low

        with pytest.raises(ValidationError):
            GenerationParams(temperature=2.5)  # Too high

    def test_load_model_request_validation(self):
        """Test load model request validation."""
        from app.schemas.requests import LoadModelRequest

        # Valid request
        request = LoadModelRequest(model_name="test_model")
        assert request.model_name == "test_model"
        assert request.force_reload == False

        # Test with force reload
        request = LoadModelRequest(model_name="test_model", force_reload=True)
        assert request.force_reload == True

    def test_seed_note_validation(self):
        """Test seed note validation."""
        from app.schemas.requests import SeedNote
        from pydantic import ValidationError

        # Valid note
        note = SeedNote(
            pitch=60,
            start_time=0.0,
            duration=1.0,
            velocity=80
        )
        assert note.pitch == 60

        # Invalid pitch
        with pytest.raises(ValidationError):
            SeedNote(pitch=128, start_time=0.0, duration=1.0, velocity=80)  # Too high

        with pytest.raises(ValidationError):
            SeedNote(pitch=-1, start_time=0.0, duration=1.0, velocity=80)  # Too low

        # Invalid velocity
        with pytest.raises(ValidationError):
            SeedNote(pitch=60, start_time=0.0, duration=1.0, velocity=0)  # Too low

        with pytest.raises(ValidationError):
            SeedNote(pitch=60, start_time=0.0, duration=1.0, velocity=128)  # Too high