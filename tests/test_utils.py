"""
Unit tests for utility functions.
"""

import pytest
import time
from unittest.mock import Mock, patch

from app.utils.helpers import (
    generate_request_id, generate_melody_id, format_file_size,
    format_duration, parse_memory_string, RateLimiter,
    PerformanceMonitor, merge_dicts, safe_cast, chunk_list
)
from app.utils.validation import (
    validate_generation_params, validate_notes_sequence,
    validate_tokens_sequence, validate_file_size, sanitize_filename
)


class TestHelpers:
    """Test cases for helper functions."""

    def test_generate_request_id(self):
        """Test request ID generation."""
        id1 = generate_request_id()
        id2 = generate_request_id()

        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert id1 != id2
        assert len(id1) == 36  # UUID length

    def test_generate_melody_id(self):
        """Test melody ID generation."""
        params = {'temperature': 1.0, 'top_k': 50}
        id1 = generate_melody_id('test_model', params)
        id2 = generate_melody_id('test_model', params)

        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert 'test_model' in id1
        # Same params should generate different IDs due to timestamp

    def test_format_file_size(self):
        """Test file size formatting."""
        assert format_file_size(0) == "0 B"
        assert format_file_size(512) == "512.0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024**2) == "1.0 MB"
        assert format_file_size(1024**3) == "1.0 GB"
        assert format_file_size(1536) == "1.5 KB"

    def test_format_duration(self):
        """Test duration formatting."""
        assert "ms" in format_duration(0.1)
        assert "s" in format_duration(1.5)
        assert "m" in format_duration(65.0)
        assert "h" in format_duration(3665.0)

    def test_parse_memory_string(self):
        """Test memory string parsing."""
        assert parse_memory_string("1GB") == 1024**3
        assert parse_memory_string("512MB") == 512 * 1024**2
        assert parse_memory_string("1024KB") == 1024**2
        assert parse_memory_string("100B") == 100

        with pytest.raises(ValueError):
            parse_memory_string("invalid")

    def test_merge_dicts(self):
        """Test dictionary merging."""
        dict1 = {'a': 1, 'b': {'c': 2}}
        dict2 = {'b': {'d': 3}, 'e': 4}

        result = merge_dicts(dict1, dict2)
        assert result['a'] == 1
        assert result['b']['c'] == 2
        assert result['b']['d'] == 3
        assert result['e'] == 4

    def test_safe_cast(self):
        """Test safe type casting."""
        assert safe_cast("123", int) == 123
        assert safe_cast("invalid", int, default=0) == 0
        assert safe_cast("1.5", float) == 1.5
        assert safe_cast(None, str, default="") == ""

    def test_chunk_list(self):
        """Test list chunking."""
        data = list(range(10))
        chunks = chunk_list(data, 3)

        assert len(chunks) == 4
        assert chunks[0] == [0, 1, 2]
        assert chunks[1] == [3, 4, 5]
        assert chunks[2] == [6, 7, 8]
        assert chunks[3] == [9]


class TestRateLimiter:
    """Test cases for RateLimiter."""

    def test_rate_limiter_basic(self):
        """Test basic rate limiting."""
        limiter = RateLimiter(max_requests=2, time_window=1)

        # First two requests should be allowed
        assert limiter.is_allowed("user1") == True
        assert limiter.is_allowed("user1") == True

        # Third request should be denied
        assert limiter.is_allowed("user1") == False

    def test_rate_limiter_different_users(self):
        """Test rate limiting with different users."""
        limiter = RateLimiter(max_requests=1, time_window=1)

        assert limiter.is_allowed("user1") == True
        assert limiter.is_allowed("user2") == True  # Different user
        assert limiter.is_allowed("user1") == False  # user1 exceeded

    def test_rate_limiter_time_window(self):
        """Test rate limiter time window reset."""
        limiter = RateLimiter(max_requests=1, time_window=1)

        assert limiter.is_allowed("user1") == True
        assert limiter.is_allowed("user1") == False

        # Wait for time window to pass
        time.sleep(1.1)
        assert limiter.is_allowed("user1") == True


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""

    def test_performance_monitor_basic(self):
        """Test basic performance monitoring."""
        monitor = PerformanceMonitor()

        # Record some requests
        monitor.record_request(success=True, response_time=0.1, model_name="test_model")
        monitor.record_request(success=False, response_time=0.2)

        metrics = monitor.get_metrics()
        assert metrics['total_requests'] == 2
        assert metrics['successful_requests'] == 1
        assert metrics['failed_requests'] == 1
        assert metrics['success_rate'] == 0.5

    def test_performance_monitor_generation(self):
        """Test generation recording."""
        monitor = PerformanceMonitor()

        monitor.record_generation(1.5, "test_model")
        monitor.record_generation(2.0, "test_model")

        metrics = monitor.get_metrics()
        assert metrics['total_generation_time'] == 3.5
        assert metrics['generation_counts']['test_model'] == 2

    def test_performance_monitor_errors(self):
        """Test error recording."""
        monitor = PerformanceMonitor()

        monitor.record_error("ValidationError")
        monitor.record_error("ValidationError")
        monitor.record_error("RuntimeError")

        metrics = monitor.get_metrics()
        assert metrics['error_counts']['ValidationError'] == 2
        assert metrics['error_counts']['RuntimeError'] == 1


class TestValidation:
    """Test cases for validation functions."""

    def test_validate_generation_params(self):
        """Test generation parameters validation."""
        # Valid params
        valid_params = {
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'max_length': 120
        }
        is_valid, error = validate_generation_params(valid_params)
        assert is_valid == True
        assert error is None

        # Invalid temperature
        invalid_params = {'temperature': 0.0}
        is_valid, error = validate_generation_params(invalid_params)
        assert is_valid == False
        assert "Temperature" in error

        # Invalid top_k
        invalid_params = {'top_k': 0}
        is_valid, error = validate_generation_params(invalid_params)
        assert is_valid == False
        assert "top_k" in error

    def test_validate_notes_sequence(self):
        """Test notes sequence validation."""
        # Valid notes
        valid_notes = [
            {'pitch': 60, 'start_time': 0.0, 'duration': 1.0, 'velocity': 80},
            {'pitch': 62, 'start_time': 1.0, 'duration': 1.0, 'velocity': 90}
        ]
        is_valid, error = validate_notes_sequence(valid_notes)
        assert is_valid == True
        assert error is None

        # Empty sequence
        is_valid, error = validate_notes_sequence([])
        assert is_valid == False
        assert "empty" in error

        # Invalid pitch
        invalid_notes = [
            {'pitch': 128, 'start_time': 0.0, 'duration': 1.0, 'velocity': 80}
        ]
        is_valid, error = validate_notes_sequence(invalid_notes)
        assert is_valid == False
        assert "pitch" in error

        # Missing field
        invalid_notes = [
            {'pitch': 60, 'start_time': 0.0, 'velocity': 80}  # Missing duration
        ]
        is_valid, error = validate_notes_sequence(invalid_notes)
        assert is_valid == False
        assert "duration" in error

    def test_validate_tokens_sequence(self):
        """Test tokens sequence validation."""
        vocab_size = 1000

        # Valid tokens
        valid_tokens = [0, 100, 200, 300, 1]
        is_valid, error = validate_tokens_sequence(valid_tokens, vocab_size)
        assert is_valid == True
        assert error is None

        # Empty sequence
        is_valid, error = validate_tokens_sequence([], vocab_size)
        assert is_valid == False
        assert "empty" in error

        # Invalid token value
        invalid_tokens = [0, 1000, 200]  # 1000 >= vocab_size
        is_valid, error = validate_tokens_sequence(invalid_tokens, vocab_size)
        assert is_valid == False
        assert "outside valid range" in error

        # Non-integer token
        invalid_tokens = [0, 1.5, 200]
        is_valid, error = validate_tokens_sequence(invalid_tokens, vocab_size)
        assert is_valid == False
        assert "integer" in error

    def test_validate_file_size(self):
        """Test file size validation."""
        # Valid size
        is_valid, error = validate_file_size(5 * 1024 * 1024, max_size_mb=10)  # 5MB
        assert is_valid == True
        assert error is None

        # Invalid size
        is_valid, error = validate_file_size(15 * 1024 * 1024, max_size_mb=10)  # 15MB
        assert is_valid == False
        assert "exceeds maximum" in error

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Valid filename
        assert sanitize_filename("test.mid") == "test.mid"

        # Invalid characters
        assert sanitize_filename("test<>file.mid") == "test__file.mid"
        assert sanitize_filename('test"file.mid') == "test_file.mid"

        # Long filename
        long_name = "a" * 300 + ".mid"
        sanitized = sanitize_filename(long_name)
        assert len(sanitized) <= 255
        assert sanitized.endswith(".mid")