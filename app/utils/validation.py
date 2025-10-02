"""
Validation utilities for API requests and responses.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import mido


def validate_midi_file(file_content: bytes) -> Tuple[bool, Optional[str]]:
    """Validate MIDI file content."""
    try:
        # Try to parse the MIDI file
        import io
        midi_file = mido.MidiFile(file=io.BytesIO(file_content))

        # Check if file has any tracks
        if not midi_file.tracks:
            return False, "MIDI file has no tracks"

        # Check if file has any note events
        has_notes = False
        for track in midi_file.tracks:
            for msg in track:
                if msg.type in ['note_on', 'note_off']:
                    has_notes = True
                    break
            if has_notes:
                break

        if not has_notes:
            return False, "MIDI file contains no note events"

        return True, None

    except Exception as e:
        return False, f"Invalid MIDI file: {str(e)}"


def validate_model_files(model_path: str, config_path: str) -> Tuple[bool, Optional[str]]:
    """Validate model and config files exist and are valid."""
    model_file = Path(model_path)
    config_file = Path(config_path)

    # Check if files exist
    if not model_file.exists():
        return False, f"Model file not found: {model_path}"

    if not config_file.exists():
        return False, f"Config file not found: {config_path}"

    # Check file extensions
    if not model_file.suffix.lower() == '.safetensors':
        return False, f"Model file must be .safetensors format: {model_path}"

    if not config_file.suffix.lower() == '.json':
        return False, f"Config file must be .json format: {config_path}"

    # Try to load config file
    try:
        import json
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        # Validate required config fields
        required_fields = ['vocab_size', 'd_model', 'n_heads', 'n_layers']
        for field in required_fields:
            if field not in config_data:
                return False, f"Config missing required field: {field}"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in config file: {e}"
    except Exception as e:
        return False, f"Error reading config file: {e}"

    return True, None


def validate_generation_params(params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate generation parameters."""
    # Temperature validation
    if 'temperature' in params:
        temp = params['temperature']
        if not isinstance(temp, (int, float)) or temp <= 0 or temp > 5.0:
            return False, "Temperature must be between 0 and 5.0"

    # Top-k validation
    if 'top_k' in params:
        top_k = params['top_k']
        if not isinstance(top_k, int) or top_k < 1 or top_k > 1000:
            return False, "top_k must be an integer between 1 and 1000"

    # Top-p validation
    if 'top_p' in params:
        top_p = params['top_p']
        if not isinstance(top_p, (int, float)) or top_p <= 0 or top_p > 1.0:
            return False, "top_p must be between 0 and 1.0"

    # Repetition penalty validation
    if 'repetition_penalty' in params:
        rep_pen = params['repetition_penalty']
        if not isinstance(rep_pen, (int, float)) or rep_pen < 0.5 or rep_pen > 3.0:
            return False, "repetition_penalty must be between 0.5 and 3.0"

    # Max length validation
    if 'max_length' in params:
        max_len = params['max_length']
        if not isinstance(max_len, int) or max_len < 1 or max_len > 2000:
            return False, "max_length must be an integer between 1 and 2000"

    return True, None


def validate_notes_sequence(notes: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Validate a sequence of musical notes."""
    if not notes:
        return False, "Notes sequence cannot be empty"

    for i, note in enumerate(notes):
        # Check required fields
        required_fields = ['pitch', 'start_time', 'duration', 'velocity']
        for field in required_fields:
            if field not in note:
                return False, f"Note {i+1} missing required field: {field}"

        # Validate pitch
        pitch = note['pitch']
        if not isinstance(pitch, int) or pitch < 0 or pitch > 127:
            return False, f"Note {i+1} pitch must be integer between 0 and 127"

        # Validate start_time
        start_time = note['start_time']
        if not isinstance(start_time, (int, float)) or start_time < 0:
            return False, f"Note {i+1} start_time must be non-negative number"

        # Validate duration
        duration = note['duration']
        if not isinstance(duration, (int, float)) or duration <= 0:
            return False, f"Note {i+1} duration must be positive number"

        # Validate velocity
        velocity = note['velocity']
        if not isinstance(velocity, int) or velocity < 1 or velocity > 127:
            return False, f"Note {i+1} velocity must be integer between 1 and 127"

    # Check for overlapping notes with same pitch
    notes_sorted = sorted(notes, key=lambda n: n['start_time'])
    for i in range(len(notes_sorted) - 1):
        current = notes_sorted[i]
        next_note = notes_sorted[i + 1]

        if (current['pitch'] == next_note['pitch'] and
            current['start_time'] + current['duration'] > next_note['start_time']):
            return False, f"Overlapping notes detected at pitch {current['pitch']}"

    return True, None


def validate_tokens_sequence(tokens: List[int], vocab_size: int) -> Tuple[bool, Optional[str]]:
    """Validate a sequence of tokens."""
    if not tokens:
        return False, "Token sequence cannot be empty"

    if len(tokens) > 1000:
        return False, "Token sequence too long (max 1000 tokens)"

    for i, token in enumerate(tokens):
        if not isinstance(token, int):
            return False, f"Token {i+1} must be an integer"

        if token < 0 or token >= vocab_size:
            return False, f"Token {i+1} ({token}) outside valid range [0, {vocab_size-1}]"

    return True, None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations."""
    import re
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + ('.' + ext if ext else '')
    return filename


def validate_file_size(file_size: int, max_size_mb: int = 10) -> Tuple[bool, Optional[str]]:
    """Validate file size."""
    max_size_bytes = max_size_mb * 1024 * 1024

    if file_size > max_size_bytes:
        return False, f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size_mb} MB)"

    return True, None


def validate_batch_request(requests: List[Dict[str, Any]], max_batch_size: int = 10) -> Tuple[bool, Optional[str]]:
    """Validate batch generation request."""
    if not requests:
        return False, "Batch request cannot be empty"

    if len(requests) > max_batch_size:
        return False, f"Batch size ({len(requests)}) exceeds maximum allowed ({max_batch_size})"

    # Validate each request
    for i, request in enumerate(requests):
        if not isinstance(request, dict):
            return False, f"Request {i+1} must be a dictionary"

        if 'model_name' not in request:
            return False, f"Request {i+1} missing required field: model_name"

    return True, None