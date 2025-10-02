"""
Request schemas for the AI Music Generation API.
Supports multiple input types and generation parameters.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class GenerationProfile(str, Enum):
    """Predefined generation profiles."""
    CREATIVE = "creative"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    EXPERIMENTAL = "experimental"
    CUSTOM = "custom"


class OutputFormat(str, Enum):
    """Output format types."""
    VST_PLUGIN = "vst_plugin"
    DAW_INTEGRATION = "daw_integration"
    RESEARCH = "research"
    MINIMAL = "minimal"


class NoteFormat(str, Enum):
    """Note timing format."""
    BEATS = "beats"
    SECONDS = "seconds"
    TICKS = "ticks"


class GenerationParams(BaseModel):
    """Generation parameters for AI models."""
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Creativity level (0.1=conservative, 2.0=very creative)"
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Limit vocabulary to top K tokens"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.01,
        le=1.0,
        description="Nucleus sampling threshold"
    )
    repetition_penalty: float = Field(
        default=1.1,
        ge=1.0,
        le=2.0,
        description="Penalty for repeating tokens"
    )
    max_length: int = Field(
        default=120,
        ge=10,
        le=500,
        description="Maximum number of tokens to generate"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation"
    )


class SeedNote(BaseModel):
    """A musical note for seeding generation."""
    pitch: int = Field(ge=0, le=127, description="MIDI pitch value")
    start_time: float = Field(ge=0.0, description="Start time (format depends on note_format)")
    duration: float = Field(gt=0.0, description="Note duration")
    velocity: int = Field(ge=1, le=127, description="MIDI velocity")


class GenerateRequest(BaseModel):
    """Main generation request with multiple input options."""

    # Model selection
    model_name: str = Field(description="Name of the model to use for generation")

    # Generation parameters
    params: Optional[GenerationParams] = Field(
        default=None,
        description="Custom generation parameters"
    )
    profile: Optional[GenerationProfile] = Field(
        default=None,
        description="Use predefined generation profile"
    )

    # Seed inputs (multiple options)
    seed_tokens: Optional[List[int]] = Field(
        default=None,
        description="Raw token sequence for seeding"
    )
    seed_notes: Optional[List[SeedNote]] = Field(
        default=None,
        description="Musical notes for seeding"
    )
    note_format: NoteFormat = Field(
        default=NoteFormat.BEATS,
        description="Format for seed_notes timing"
    )

    # Generation options
    num_generations: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of variations to generate"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.VST_PLUGIN,
        description="Output format configuration"
    )

    # Additional options
    include_seed_in_output: bool = Field(
        default=False,
        description="Include seed content in the output"
    )
    tempo: Optional[float] = Field(
        default=120.0,
        ge=60.0,
        le=200.0,
        description="Target tempo for MIDI output"
    )
    time_signature: Optional[List[int]] = Field(
        default=[4, 4],
        description="Time signature [numerator, denominator]"
    )

    @field_validator('time_signature')
    @classmethod
    def validate_time_signature(cls, v):
        if v and len(v) != 2:
            raise ValueError("Time signature must be [numerator, denominator]")
        if v and (v[0] < 1 or v[1] not in [1, 2, 4, 8, 16]):
            raise ValueError("Invalid time signature values")
        return v


class LoadModelRequest(BaseModel):
    """Request to load a specific model."""
    model_name: str = Field(description="Name of the model to load")
    force_reload: bool = Field(
        default=False,
        description="Force reload even if already loaded"
    )


class UnloadModelRequest(BaseModel):
    """Request to unload a model."""
    model_name: str = Field(description="Name of the model to unload")


class BatchGenerateRequest(BaseModel):
    """Batch generation request for multiple models or parameters."""
    requests: List[GenerateRequest] = Field(
        description="List of generation requests"
    )
    parallel: bool = Field(
        default=True,
        description="Execute requests in parallel"
    )

    @field_validator('requests')
    @classmethod
    def validate_batch_size(cls, v):
        if len(v) > 20:  # Reasonable limit
            raise ValueError("Batch size cannot exceed 20 requests")
        return v


class GenerateFromMidiRequest(BaseModel):
    """Generate using MIDI file as seed (for form data)."""
    model_name: str = Field(description="Name of the model to use")
    params: Optional[str] = Field(
        default=None,
        description="JSON string of GenerationParams"
    )
    profile: Optional[GenerationProfile] = Field(
        default=None,
        description="Generation profile to use"
    )
    num_generations: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of variations"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.VST_PLUGIN,
        description="Output format"
    )
    use_full_midi: bool = Field(
        default=False,
        description="Use entire MIDI file or just extract melody"
    )
    max_seed_length: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum tokens to use from MIDI seed"
    )


class ConfigUpdateRequest(BaseModel):
    """Request to update API configuration."""
    reload_config: bool = Field(
        default=False,
        description="Reload configuration from file"
    )
    update_storage_settings: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Update storage settings"
    )


# Query parameter models for simple endpoints
class SimpleGenerateParams(BaseModel):
    """Simple generation parameters for GET endpoints."""
    model_name: str
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    max_length: int = 120
    num_generations: int = 1
    profile: Optional[str] = None