"""
Response schemas for the AI Music Generation API.
Comprehensive output formats for different use cases.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class GenerationStatus(str, Enum):
    """Status of generation request."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"


class ModelStatus(str, Enum):
    """Model loading status."""
    LOADED = "loaded"
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    ERROR = "error"


class NoteResponse(BaseModel):
    """Musical note in response format."""
    pitch: int = Field(description="MIDI pitch value (0-127)")
    start_time: float = Field(description="Start time in specified format")
    duration: float = Field(description="Note duration")
    velocity: int = Field(description="MIDI velocity (1-127)")
    channel: int = Field(default=0, description="MIDI channel")


class GenerationMetadata(BaseModel):
    """Metadata about the generation process."""
    model_name: str = Field(description="Model used for generation")
    model_type: str = Field(description="Type of model")
    generation_params: Dict[str, Any] = Field(description="Parameters used")
    generation_time: float = Field(description="Time taken to generate (seconds)")
    token_count: int = Field(description="Number of tokens generated")
    note_count: int = Field(description="Number of musical notes")
    seed_used: bool = Field(description="Whether seed input was used")
    seed_type: Optional[str] = Field(description="Type of seed used")
    duration_beats: float = Field(description="Total duration in beats")
    duration_seconds: Optional[float] = Field(description="Total duration in seconds")
    stopped_early: bool = Field(description="Whether generation stopped on EOS token")


class GeneratedMelody(BaseModel):
    """A single generated melody."""
    id: str = Field(description="Unique generation ID")
    tokens: Optional[List[int]] = Field(description="Raw token sequence")
    notes: List[NoteResponse] = Field(description="Musical notes")
    midi_base64: Optional[str] = Field(description="MIDI file as base64")
    midi_url: Optional[str] = Field(description="URL to download MIDI file")
    metadata: GenerationMetadata = Field(description="Generation metadata")

    # Advanced outputs for research
    attention_weights: Optional[List[List[float]]] = Field(
        description="Attention weights (if requested)"
    )
    token_probabilities: Optional[List[Dict[int, float]]] = Field(
        description="Token probabilities (if requested)"
    )


class GenerateResponse(BaseModel):
    """Main generation response."""
    status: GenerationStatus = Field(description="Overall generation status")
    request_id: str = Field(description="Unique request identifier")
    timestamp: str = Field(description="Response timestamp")

    # Results
    melodies: List[GeneratedMelody] = Field(description="Generated melodies")
    success_count: int = Field(description="Number of successful generations")
    total_requested: int = Field(description="Total number requested")

    # Timing and performance
    total_generation_time: float = Field(description="Total time for all generations")
    average_generation_time: float = Field(description="Average time per generation")

    # Model information
    model_info: Dict[str, Any] = Field(description="Information about the model used")

    # Error information (if any)
    errors: Optional[List[str]] = Field(description="Any errors that occurred")
    warnings: Optional[List[str]] = Field(description="Any warnings")


class ModelInfo(BaseModel):
    """Information about a model."""
    name: str = Field(description="Model name")
    type: str = Field(description="Model type")
    description: str = Field(description="Model description")
    tags: List[str] = Field(description="Model tags")

    # Architecture info
    vocab_size: int = Field(description="Vocabulary size")
    parameter_count: Optional[int] = Field(description="Number of parameters")

    # File info
    model_file: str = Field(description="Path to model file")
    config_file: str = Field(description="Path to config file")
    file_exists: bool = Field(description="Whether files exist")
    supported: bool = Field(description="Whether model type is supported")

    # Runtime info
    status: ModelStatus = Field(description="Current loading status")
    load_time: Optional[str] = Field(description="When model was loaded")
    last_accessed: Optional[str] = Field(description="Last access time")
    access_count: Optional[int] = Field(description="Number of times accessed")


class ModelsListResponse(BaseModel):
    """Response listing all available models."""
    models: Dict[str, ModelInfo] = Field(description="Available models")
    total_models: int = Field(description="Total number of models")
    loaded_models: int = Field(description="Number of currently loaded models")
    supported_types: List[str] = Field(description="Supported model types")


class ModelLoadResponse(BaseModel):
    """Response for model loading operations."""
    success: bool = Field(description="Whether operation succeeded")
    message: str = Field(description="Status message")
    model_name: str = Field(description="Model name")

    # Success details
    load_time: Optional[float] = Field(description="Time taken to load")
    model_type: Optional[str] = Field(description="Type of model")
    parameter_count: Optional[int] = Field(description="Number of parameters")
    vocab_size: Optional[int] = Field(description="Vocabulary size")
    device: Optional[str] = Field(description="Device model is running on")

    # Error details
    error: Optional[str] = Field(description="Error message if failed")
    already_loaded: Optional[bool] = Field(description="Whether model was already loaded")


class SystemStatusResponse(BaseModel):
    """System status and health information."""
    status: str = Field(description="Overall system status")
    timestamp: str = Field(description="Current timestamp")
    uptime: Optional[float] = Field(description="System uptime in seconds")

    # Model information
    loaded_models: List[str] = Field(description="Currently loaded models")
    max_loaded_models: int = Field(description="Maximum allowed loaded models")

    # Memory information
    memory_info: Dict[str, Any] = Field(description="Memory usage information")

    # Performance metrics
    total_generations: Optional[int] = Field(description="Total generations served")
    average_response_time: Optional[float] = Field(description="Average response time")

    # Configuration
    config_loaded: bool = Field(description="Whether configuration is loaded")
    config_file: str = Field(description="Configuration file path")


class BatchGenerateResponse(BaseModel):
    """Response for batch generation requests."""
    status: GenerationStatus = Field(description="Overall batch status")
    request_id: str = Field(description="Batch request ID")
    timestamp: str = Field(description="Response timestamp")

    # Results
    results: List[GenerateResponse] = Field(description="Individual generation results")
    success_count: int = Field(description="Number of successful requests")
    total_requests: int = Field(description="Total number of requests")

    # Timing
    total_batch_time: float = Field(description="Total time for batch")
    parallel_execution: bool = Field(description="Whether executed in parallel")

    # Errors
    errors: Optional[List[str]] = Field(description="Batch-level errors")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(description="Error message")
    error_code: Optional[str] = Field(description="Error code")
    detail: Optional[str] = Field(description="Detailed error information")
    timestamp: str = Field(description="Error timestamp")
    request_id: Optional[str] = Field(description="Request ID if available")
    suggestions: Optional[List[str]] = Field(description="Suggested solutions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Health status")
    timestamp: str = Field(description="Check timestamp")
    version: str = Field(description="API version")
    dependencies: Dict[str, str] = Field(description="Dependency status")


class ConfigResponse(BaseModel):
    """Configuration information response."""
    loaded: bool = Field(description="Whether config is loaded")
    file_path: str = Field(description="Configuration file path")
    last_modified: Optional[str] = Field(description="Last modification time")

    # Summary information
    total_models: int = Field(description="Total configured models")
    total_tokenizers: int = Field(description="Total configured tokenizers")
    generation_profiles: List[str] = Field(description="Available generation profiles")
    output_formats: List[str] = Field(description="Available output formats")

    # Settings summary
    storage_settings: Dict[str, Any] = Field(description="Storage configuration")
    performance_settings: Dict[str, Any] = Field(description="Performance configuration")
    security_settings: Dict[str, Any] = Field(description="Security configuration")