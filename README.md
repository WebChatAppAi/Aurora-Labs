# 🎵 AI Music Generation API

**Professional, scalable FastAPI server for AI-powered music generation with multi-model support, YAML configuration, and comprehensive VST plugin integration.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🚀 Features

- **🎯 Multiple AI Model Support**: Melody, harmony, drum pattern generators
- **⚙️ YAML Configuration**: Centralized model and parameter management
- **🎛️ Flexible Generation**: Multiple input types (MIDI files, note sequences, tokens)
- **📱 VST Plugin Ready**: Optimized output formats for music software
- **🔄 Dynamic Model Loading**: Load/unload models on demand
- **📊 Performance Monitoring**: Built-in metrics and rate limiting
- **🧪 Comprehensive Testing**: Full unit test coverage
- **📚 Interactive Documentation**: Auto-generated API docs

## 🎼 Use Cases

### 1. **VST Plugin Integration**
```python
# Perfect for FL Studio, Ableton Live, Logic Pro plugins
response = requests.post("http://localhost:8000/api/v1/generate", json={
    "model_name": "melody_small",
    "output_format": "vst_plugin",
    "params": {
        "temperature": 1.2,
        "max_length": 100
    }
})

# Get notes directly for piano roll
notes = response.json()["melodies"][0]["notes"]
# Each note: {pitch, start_time, duration, velocity}
```

### 2. **MIDI-to-MIDI Enhancement**
```python
# Upload existing MIDI, get AI variations
files = {"midi_file": open("input.mid", "rb")}
data = {"model_name": "melody_large", "num_generations": 3}

response = requests.post("http://localhost:8000/api/v1/generate/from-midi",
                        files=files, data=data)

# Get enhanced MIDI files
for melody in response.json()["melodies"]:
    midi_data = base64.b64decode(melody["midi_base64"])
    with open(f"enhanced_{melody['id']}.mid", "wb") as f:
        f.write(midi_data)
```

### 3. **Real-time Music Composition**
```python
# Quick generation for live performance
response = requests.get("http://localhost:8000/api/v1/generate/simple", params={
    "model_name": "melody_small",
    "temperature": 1.5,  # High creativity
    "max_length": 64,    # Short phrases
    "num_generations": 5
})
```

### 4. **Batch Processing for Albums**
```python
# Generate multiple tracks with different styles
batch_request = {
    "requests": [
        {"model_name": "melody_large", "profile": "creative"},
        {"model_name": "harmony_basic", "profile": "balanced"},
        {"model_name": "drum_patterns", "profile": "experimental"}
    ],
    "parallel": True
}

response = requests.post("http://localhost:8000/api/v1/generate/batch",
                        json=batch_request)
```

### 5. **Music Education & Research**
```python
# Get detailed analysis with attention weights
response = requests.post("http://localhost:8000/api/v1/generate", json={
    "model_name": "melody_large",
    "output_format": "research",
    "params": {"temperature": 1.0}
})

# Access model internals
melody = response.json()["melodies"][0]
attention_weights = melody["attention_weights"]
token_probabilities = melody["token_probabilities"]
```

## 📦 Installation

### Prerequisites
- Python 3.9+
- GPU with CUDA (optional, for faster generation)
- Your trained AI music models (.safetensors format)

### Quick Setup
```bash
# Clone repository
git clone <your-repo>
cd ALV-MODEL-LATEST-API

# Install dependencies
pip install -r requirements.txt

# Configure models (edit config/models.yaml)
cp config/models.yaml.example config/models.yaml
# Edit paths to your model files

# Start server
python start_server.py --port 8000
```

### Docker Setup
```bash
# Build image
docker build -t ai-music-api .

# Run container
docker run -p 8000:8000 -v ./models:/app/models -v ./config:/app/config ai-music-api
```

## ⚙️ Configuration

Edit `config/models.yaml` to define your models:

```yaml
storage:
  models_root: "./models"  # Your model directory
  max_loaded_models: 3

models:
  my_melody_model:
    type: "MelodyTransformer"
    model_file: "my_model/model.safetensors"
    config_file: "my_model/config.json"
    architecture:
      vocab_size: 901
      d_model: 1024
      # ... other parameters
    tags: ["melody", "custom"]
    description: "My custom melody generator"
```

## 🔌 API Usage

### Start the Server
```bash
python start_server.py --port 8000
# API docs: http://localhost:8000/docs
```

### Basic Generation
```bash
# Load a model
curl -X POST "http://localhost:8000/api/v1/models/load" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "melody_small"}'

# Generate music
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "melody_small",
    "params": {
      "temperature": 1.2,
      "max_length": 100
    },
    "num_generations": 1
  }'
```

### Generation with MIDI Seed
```bash
curl -X POST "http://localhost:8000/api/v1/generate/from-midi" \
  -F "model_name=melody_small" \
  -F "midi_file=@input.mid" \
  -F "num_generations=3"
```

## 📊 Model Management

### List Available Models
```bash
curl "http://localhost:8000/api/v1/models"
```

### System Status
```bash
curl "http://localhost:8000/api/v1/status"
```

### Performance Monitoring
The API includes built-in monitoring:
- Request/response times
- Model loading times
- Memory usage tracking
- Error rate monitoring

## 🎛️ Generation Parameters

| Parameter | Range | Description |
|-----------|--------|-------------|
| `temperature` | 0.1-2.0 | Creativity level (higher = more creative) |
| `top_k` | 1-200 | Vocabulary limit for sampling |
| `top_p` | 0.01-1.0 | Nucleus sampling threshold |
| `repetition_penalty` | 1.0-2.0 | Penalty for repeating patterns |
| `max_length` | 10-500 | Maximum tokens to generate |

## 🔧 Development

### Run Tests
```bash
# All tests
pytest

# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# With coverage
pytest --cov=app
```

### Development Server
```bash
python start_server.py --reload --port 8000
```

### Add New Model Type
1. Create model class in `app/models/`
2. Register factory in `app/services/model_registry.py`
3. Update YAML schema in `app/core/config.py`
4. Add tests in `tests/`

## 📚 Documentation

- **API Reference**: [api-usage.md](api-usage.md)
- **Development Guide**: [dev-notes.md](dev-notes.md)
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🚀 Deployment

### Production Setup
```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Variables
```bash
export CONFIG_PATH="config/production.yaml"
export LOG_LEVEL="INFO"
export MAX_WORKERS="4"
```

### Load Balancing
For high-traffic deployments, use multiple instances behind a load balancer:
```nginx
upstream ai_music_api {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Read [dev-notes.md](dev-notes.md) for architecture details
4. Add tests for new features
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

## 📈 Performance Tips

1. **GPU Usage**: Models automatically use CUDA if available
2. **Memory Management**: Configure `max_loaded_models` based on your RAM/VRAM
3. **Caching**: Enable model caching for faster subsequent loads
4. **Batch Processing**: Use batch endpoints for multiple generations

## 🐛 Troubleshooting

### Common Issues

**Models not loading:**
```bash
# Check model files exist
ls -la models/your_model/

# Validate configuration
python -c "from app.core.config import get_config; print(get_config().models)"
```

**Out of memory:**
```yaml
# Reduce max_loaded_models in config/models.yaml
storage:
  max_loaded_models: 1
```

**Slow generation:**
- Ensure GPU is being used (`nvidia-smi`)
- Reduce `max_length` parameter
- Use smaller models for real-time applications

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Model architecture based on Transformer research
- MIDI processing with [mido](https://mido.readthedocs.io/)
- Configuration management with [Pydantic](https://pydantic-docs.helpmanual.io/)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **API Docs**: http://localhost:8000/docs

---

**Made with ❤️ for the music production community**