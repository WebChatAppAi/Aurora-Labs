"""
General helper utilities for the API.
"""

import uuid
import time
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def generate_request_id() -> str:
    """Generate unique request ID."""
    return str(uuid.uuid4())


def generate_melody_id(model_name: str, params: Dict[str, Any]) -> str:
    """Generate unique melody ID based on model and parameters."""
    # Create hash of parameters for reproducibility
    params_str = str(sorted(params.items()))
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}_{params_hash}"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def calculate_model_memory_usage(parameter_count: int, precision: str = "float32") -> int:
    """Estimate model memory usage in bytes."""
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "int8": 1
    }

    multiplier = bytes_per_param.get(precision, 4)

    # Base model memory
    model_memory = parameter_count * multiplier

    # Add overhead for gradients, optimizer states, etc. (approximate)
    overhead_factor = 1.2

    return int(model_memory * overhead_factor)


def parse_memory_string(memory_str: str) -> int:
    """Parse memory string like '1GB', '512MB' to bytes."""
    memory_str = memory_str.upper().strip()

    # Order matters - check longer units first to avoid 'B' matching 'GB'
    multipliers = [
        ('TB', 1024**4),
        ('GB', 1024**3),
        ('MB', 1024**2),
        ('KB', 1024),
        ('B', 1)
    ]

    for unit, multiplier in multipliers:
        if memory_str.endswith(unit):
            try:
                value = float(memory_str[:-len(unit)])
                return int(value * multiplier)
            except ValueError:
                break

    raise ValueError(f"Invalid memory format: {memory_str}")


class RateLimiter:
    """Simple rate limiter for API endpoints."""

    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.requests: Dict[str, List[float]] = {}

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for given identifier."""
        now = time.time()

        if identifier not in self.requests:
            self.requests[identifier] = []

        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.time_window
        ]

        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False

        # Add current request
        self.requests[identifier].append(now)
        return True

    def get_reset_time(self, identifier: str) -> Optional[float]:
        """Get time when rate limit resets for identifier."""
        if identifier not in self.requests or not self.requests[identifier]:
            return None

        oldest_request = min(self.requests[identifier])
        return oldest_request + self.time_window


class PerformanceMonitor:
    """Monitor API performance metrics."""

    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_generation_time': 0.0,
            'average_response_time': 0.0,
            'model_load_times': {},
            'generation_counts': {},
            'error_counts': {}
        }
        self.start_time = time.time()

    def record_request(self, success: bool, response_time: float, model_name: str = None):
        """Record a request."""
        self.metrics['total_requests'] += 1

        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1

        # Update average response time
        total_time = self.metrics['average_response_time'] * (self.metrics['total_requests'] - 1)
        self.metrics['average_response_time'] = (total_time + response_time) / self.metrics['total_requests']

        if model_name:
            self.metrics['generation_counts'][model_name] = \
                self.metrics['generation_counts'].get(model_name, 0) + 1

    def record_generation(self, generation_time: float, model_name: str):
        """Record a generation operation."""
        self.metrics['total_generation_time'] += generation_time

        if model_name not in self.metrics['generation_counts']:
            self.metrics['generation_counts'][model_name] = 0
        self.metrics['generation_counts'][model_name] += 1

    def record_model_load(self, model_name: str, load_time: float):
        """Record model loading time."""
        self.metrics['model_load_times'][model_name] = load_time

    def record_error(self, error_type: str):
        """Record an error."""
        self.metrics['error_counts'][error_type] = \
            self.metrics['error_counts'].get(error_type, 0) + 1

    def get_uptime(self) -> float:
        """Get API uptime in seconds."""
        return time.time() - self.start_time

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        metrics = self.metrics.copy()
        metrics['uptime'] = self.get_uptime()
        metrics['success_rate'] = (
            self.metrics['successful_requests'] / max(1, self.metrics['total_requests'])
        )
        return metrics


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def safe_cast(value: Any, target_type: type, default: Any = None) -> Any:
    """Safely cast value to target type with default fallback."""
    try:
        # Handle None values specifically
        if value is None:
            return default
        return target_type(value)
    except (ValueError, TypeError):
        return default


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def create_error_context(error: Exception, additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create comprehensive error context for logging."""
    context = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.now().isoformat()
    }

    if additional_info:
        context.update(additional_info)

    return context


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor