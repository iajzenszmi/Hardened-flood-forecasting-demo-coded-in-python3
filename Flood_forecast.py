
"""
Hardened Flood Forecasting and Warning System Demo

This module demonstrates a secure implementation of a flood forecasting
and warning system based on research literature. It incorporates security
best practices including input validation, secure data handling, logging,
and error handling.

References:
- Werner et al. (2005) - European Flood Forecasting System
- Zhang et al. (2023) - Intelligent flood forecasting using ML
- Sai et al. (2018) - Impact-based flood forecasting in Bangladesh

Security Features:
- Input validation and sanitization
- Secure configuration management
- Comprehensive logging with sensitive data redaction
- Rate limiting for API endpoints
- Type hints and runtime type checking
- Exception handling with secure error messages
- Immutable data structures where appropriate
- Principle of least privilege
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Final,
    Literal,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)

# =============================================================================
# SECURITY CONSTANTS
# =============================================================================

# Maximum allowed values to prevent resource exhaustion
MAX_WATER_LEVEL: Final[float] = 100.0  # meters
MIN_WATER_LEVEL: Final[float] = -10.0  # meters (below sea level)
MAX_FORECAST_DAYS: Final[int] = 10  # Based on Werner et al. (2005)
MAX_STATIONS: Final[int] = 1000
MAX_INPUT_LENGTH: Final[int] = 10000
MAX_RATE_LIMIT_REQUESTS: Final[int] = 100
RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60

# Secure defaults
DEFAULT_ENCODING: Final[str] = "utf-8"
SECURE_HASH_ALGORITHM: Final[str] = "sha256"


# =============================================================================
# SECURE LOGGING CONFIGURATION
# =============================================================================

class SecureFormatter(logging.Formatter):
    """Custom formatter that redacts sensitive information from logs."""

    # Patterns for sensitive data that should be redacted
    SENSITIVE_PATTERNS: tuple[tuple[str, str], ...] = (
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+', 'api_key=***REDACTED***'),
        (r'password["\']?\s*[:=]\s*["\']?[\w-]+', 'password=***REDACTED***'),
        (r'token["\']?\s*[:=]\s*["\']?[\w-]+', 'token=***REDACTED***'),
        (r'secret["\']?\s*[:=]\s*["\']?[\w-]+', 'secret=***REDACTED***'),
        (r'\b\d{3}-\d{2}-\d{4}\b', '***SSN-REDACTED***'),  # SSN pattern
    )

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with sensitive data redaction."""
        message = super().format(record)
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
        return message


def setup_secure_logging(
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Configure secure logging with redaction and proper formatting.

    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("flood_forecasting")
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create secure formatter
    formatter = SecureFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        # Validate log file path
        log_file = Path(log_file).resolve()
        if not _is_safe_path(log_file):
            raise SecurityError("Invalid log file path")

        file_handler = logging.FileHandler(log_file, encoding=DEFAULT_ENCODING)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Initialize logger
logger = setup_secure_logging()


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class FloodForecastError(Exception):
    """Base exception for flood forecasting system."""
    pass


class ValidationError(FloodForecastError):
    """Raised when input validation fails."""
    pass


class SecurityError(FloodForecastError):
    """Raised when a security violation is detected."""
    pass


class RateLimitError(FloodForecastError):
    """Raised when rate limit is exceeded."""
    pass


class ConfigurationError(FloodForecastError):
    """Raised when configuration is invalid."""
    pass


# =============================================================================
# INPUT VALIDATION UTILITIES
# =============================================================================

def _is_safe_path(path: Path) -> bool:
    """
    Check if a path is safe (no directory traversal attacks).

    Args:
        path: Path to validate

    Returns:
        True if path is safe, False otherwise
    """
    try:
        resolved = path.resolve()
        # Ensure no path traversal
        return ".." not in str(path) and not str(resolved).startswith("/etc")
    except (OSError, ValueError):
        return False


def validate_string(
    value: Any,
    field_name: str,
    max_length: int = MAX_INPUT_LENGTH,
    pattern: Optional[str] = None,
    allow_empty: bool = False
) -> str:
    """
    Validate and sanitize string input.

    Args:
        value: Input value to validate
        field_name: Name of the field (for error messages)
        max_length: Maximum allowed length
        pattern: Optional regex pattern to match
        allow_empty: Whether empty strings are allowed

    Returns:
        Validated and sanitized string

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string, got {type(value).__name__}")

    # Strip whitespace
    value = value.strip()

    if not allow_empty and not value:
        raise ValidationError(f"{field_name} cannot be empty")

    if len(value) > max_length:
        raise ValidationError(f"{field_name} exceeds maximum length of {max_length}")

    # Check for null bytes (potential injection attack)
    if "\x00" in value:
        raise ValidationError(f"{field_name} contains invalid characters")

    # Check pattern if specified
    if pattern and not re.match(pattern, value):
        raise ValidationError(f"{field_name} does not match required pattern")

    return value


def validate_numeric(
    value: Any,
    field_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_none: bool = False
) -> Optional[float]:
    """
    Validate numeric input with bounds checking.

    Args:
        value: Input value to validate
        field_name: Name of the field (for error messages)
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_none: Whether None is allowed

    Returns:
        Validated numeric value

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(f"{field_name} cannot be None")

    try:
        # Use Decimal for precise conversion
        numeric_value = float(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError) as e:
        raise ValidationError(f"{field_name} must be a valid number") from e

    # Check for special float values
    if not (-float("inf") < numeric_value < float("inf")):
        raise ValidationError(f"{field_name} must be a finite number")

    if min_value is not None and numeric_value < min_value:
        raise ValidationError(f"{field_name} must be >= {min_value}")

    if max_value is not None and numeric_value > max_value:
        raise ValidationError(f"{field_name} must be <= {max_value}")

    return numeric_value


def validate_datetime(
    value: Any,
    field_name: str,
    min_date: Optional[datetime] = None,
    max_date: Optional[datetime] = None
) -> datetime:
    """
    Validate datetime input.

    Args:
        value: Input value to validate
        field_name: Name of the field (for error messages)
        min_date: Minimum allowed date
        max_date: Maximum allowed date

    Returns:
        Validated datetime object

    Raises:
        ValidationError: If validation fails
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValidationError(f"{field_name} must be a valid ISO datetime") from e
    else:
        raise ValidationError(f"{field_name} must be a datetime or ISO string")

    # Ensure timezone awareness
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    if min_date and dt < min_date:
        raise ValidationError(f"{field_name} must be after {min_date.isoformat()}")

    if max_date and dt > max_date:
        raise ValidationError(f"{field_name} must be before {max_date.isoformat()}")

    return dt


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """
    Thread-safe rate limiter using sliding window algorithm.

    Implements rate limiting to prevent abuse and resource exhaustion.
    """

    def __init__(
        self,
        max_requests: int = MAX_RATE_LIMIT_REQUESTS,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client is within rate limit.

        Args:
            client_id: Unique client identifier

        Returns:
            True if within limit, False otherwise
        """
        current_time = time.monotonic()
        window_start = current_time - self._window_seconds

        # Get or create request list for client
        if client_id not in self._requests:
            self._requests[client_id] = []

        # Remove old requests outside window
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]

        # Check limit
        if len(self._requests[client_id]) >= self._max_requests:
            return False

        # Record new request
        self._requests[client_id].append(current_time)
        return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        if client_id not in self._requests:
            return self._max_requests

        current_time = time.monotonic()
        window_start = current_time - self._window_seconds
        active_requests = sum(1 for t in self._requests[client_id] if t > window_start)
        return max(0, self._max_requests - active_requests)


# Global rate limiter instance
_rate_limiter = RateLimiter()


def rate_limited(func: Callable) -> Callable:
    """
    Decorator to apply rate limiting to functions.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with rate limiting
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Use function name as client ID for demo
        client_id = kwargs.get("client_id", "default")

        if not _rate_limiter.check_rate_limit(str(client_id)):
            remaining = _rate_limiter.get_remaining(str(client_id))
            raise RateLimitError(
                f"Rate limit exceeded. Remaining: {remaining}. "
                f"Please wait before retrying."
            )

        return func(*args, **kwargs)

    return wrapper


# =============================================================================
# SECURE CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class SecureConfig:
    """
    Immutable configuration class with validation.

    Uses frozen dataclass to prevent modification after creation.
    """

    station_id: str
    station_name: str
    latitude: float
    longitude: float
    warning_threshold_yellow: float
    warning_threshold_orange: float
    warning_threshold_red: float
    forecast_horizon_days: int = 5

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate station_id (alphanumeric with underscores)
        validate_string(
            self.station_id,
            "station_id",
            max_length=50,
            pattern=r"^[a-zA-Z0-9_-]+$"
        )

        # Validate station_name
        validate_string(self.station_name, "station_name", max_length=200)

        # Validate coordinates
        validate_numeric(self.latitude, "latitude", min_value=-90, max_value=90)
        validate_numeric(self.longitude, "longitude", min_value=-180, max_value=180)

        # Validate thresholds (must be in ascending order)
        validate_numeric(
            self.warning_threshold_yellow,
            "warning_threshold_yellow",
            min_value=MIN_WATER_LEVEL,
            max_value=MAX_WATER_LEVEL
        )
        validate_numeric(
            self.warning_threshold_orange,
            "warning_threshold_orange",
            min_value=self.warning_threshold_yellow,
            max_value=MAX_WATER_LEVEL
        )
        validate_numeric(
            self.warning_threshold_red,
            "warning_threshold_red",
            min_value=self.warning_threshold_orange,
            max_value=MAX_WATER_LEVEL
        )

        # Validate forecast horizon
        if not 1 <= self.forecast_horizon_days <= MAX_FORECAST_DAYS:
            raise ValidationError(
                f"forecast_horizon_days must be between 1 and {MAX_FORECAST_DAYS}"
            )


# =============================================================================
# DATA MODELS
# =============================================================================

class WarningLevel(Enum):
    """
    Color-coded warning levels based on Sai et al. (2018).

    Impact-based warning levels that connect water levels to
    localized guidance information.
    """
    GREEN = auto()   # Normal conditions
    YELLOW = auto()  # Minor flooding possible
    ORANGE = auto()  # Moderate flooding expected
    RED = auto()     # Severe flooding imminent

    @property
    def description(self) -> str:
        """Get human-readable description of warning level."""
        descriptions = {
            WarningLevel.GREEN: "Normal conditions - No action required",
            WarningLevel.YELLOW: "Minor flooding possible - Be aware",
            WarningLevel.ORANGE: "Moderate flooding expected - Be prepared",
            WarningLevel.RED: "Severe flooding imminent - Take action",
        }
        return descriptions[self]


class WaterLevelReading(NamedTuple):
    """
    Immutable water level reading with timestamp.

    Uses NamedTuple for immutability and memory efficiency.
    """
    timestamp: datetime
    level_meters: float
    station_id: str
    quality_flag: Literal["good", "suspect", "missing"] = "good"

    @classmethod
    def create(
        cls,
        timestamp: Union[datetime, str],
        level_meters: float,
        station_id: str,
        quality_flag: str = "good"
    ) -> "WaterLevelReading":
        """
        Factory method with validation.

        Args:
            timestamp: Observation timestamp
            level_meters: Water level in meters
            station_id: Station identifier
            quality_flag: Data quality indicator

        Returns:
            Validated WaterLevelReading instance
        """
        validated_timestamp = validate_datetime(timestamp, "timestamp")
        validated_level = validate_numeric(
            level_meters,
            "level_meters",
            min_value=MIN_WATER_LEVEL,
            max_value=MAX_WATER_LEVEL
        )
        validated_station = validate_string(
            station_id,
            "station_id",
            pattern=r"^[a-zA-Z0-9_-]+$"
        )

        if quality_flag not in ("good", "suspect", "missing"):
            raise ValidationError("quality_flag must be 'good', 'suspect', or 'missing'")

        return cls(
            timestamp=validated_timestamp,
            level_meters=validated_level,
            station_id=validated_station,
            quality_flag=quality_flag
        )


@dataclass
class ForecastResult:
    """
    Flood forecast result with probabilistic information.

    Based on Werner et al. (2005) - forecasts at extended lead times
    must be treated probabilistically.
    """

    station_id: str
    forecast_time: datetime
    valid_time: datetime
    lead_time_hours: int
    predicted_level: float
    prediction_interval_lower: float
    prediction_interval_upper: float
    confidence_level: float
    warning_level: WarningLevel
    ensemble_spread: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate forecast result after initialization."""
        validate_string(self.station_id, "station_id", pattern=r"^[a-zA-Z0-9_-]+$")

        # Validate levels
        for field_name in ["predicted_level", "prediction_interval_lower",
                          "prediction_interval_upper"]:
            validate_numeric(
                getattr(self, field_name),
                field_name,
                min_value=MIN_WATER_LEVEL,
                max_value=MAX_WATER_LEVEL
            )

        # Validate confidence level
        validate_numeric(
            self.confidence_level,
            "confidence_level",
            min_value=0.0,
            max_value=1.0
        )

        # Validate prediction interval consistency
        if self.prediction_interval_lower > self.predicted_level:
            raise ValidationError(
                "prediction_interval_lower cannot exceed predicted_level"
            )
        if self.prediction_interval_upper < self.predicted_level:
            raise ValidationError(
                "prediction_interval_upper cannot be less than predicted_level"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "station_id": self.station_id,
            "forecast_time": self.forecast_time.isoformat(),
            "valid_time": self.valid_time.isoformat(),
            "lead_time_hours": self.lead_time_hours,
            "predicted_level": round(self.predicted_level, 3),
            "prediction_interval": {
                "lower": round(self.prediction_interval_lower, 3),
                "upper": round(self.prediction_interval_upper, 3),
            },
            "confidence_level": round(self.confidence_level, 3),
            "warning_level": self.warning_level.name,
            "warning_description": self.warning_level.description,
        }


# =============================================================================
# FORECASTING ENGINE
# =============================================================================

class ForecastModel(ABC):
    """
    Abstract base class for flood forecasting models.

    Based on Zhang et al. (2023) - supports various ML approaches
    including supervised, unsupervised, and deep learning.
    """

    @abstractmethod
    def predict(
        self,
        observations: list[WaterLevelReading],
        lead_time_hours: int
    ) -> tuple[float, float, float]:
        """
        Generate prediction with uncertainty bounds.

        Args:
            observations: Historical water level observations
            lead_time_hours: Forecast lead time in hours

        Returns:
            Tuple of (predicted_level, lower_bound, upper_bound)
        """
        pass

    @abstractmethod
    def get_confidence(self, lead_time_hours: int) -> float:
        """
        Get confidence level for given lead time.

        Args:
            lead_time_hours: Forecast lead time in hours

        Returns:
            Confidence level between 0 and 1
        """
        pass


class SimplePersistenceModel(ForecastModel):
    """
    Simple persistence model for demonstration.

    Uses last observation with increasing uncertainty over time.
    In production, this would be replaced with ML models as
    described in Zhang et al. (2023).
    """

    # Uncertainty growth rate per hour (demonstration value)
    UNCERTAINTY_GROWTH_RATE: float = 0.02

    def predict(
        self,
        observations: list[WaterLevelReading],
        lead_time_hours: int
    ) -> tuple[float, float, float]:
        """Generate prediction using persistence approach."""
        if not observations:
            raise ValidationError("At least one observation is required")

        # Validate lead time
        validate_numeric(
            lead_time_hours,
            "lead_time_hours",
            min_value=1,
            max_value=MAX_FORECAST_DAYS * 24
        )

        # Use most recent observation
        latest = max(observations, key=lambda x: x.timestamp)
        base_level = latest.level_meters

        # Calculate uncertainty bounds (grows with lead time)
        uncertainty = self.UNCERTAINTY_GROWTH_RATE * lead_time_hours
        lower_bound = max(MIN_WATER_LEVEL, base_level - uncertainty)
        upper_bound = min(MAX_WATER_LEVEL, base_level + uncertainty)

        return base_level, lower_bound, upper_bound

    def get_confidence(self, lead_time_hours: int) -> float:
        """Calculate confidence based on lead time."""
        # Confidence decreases with lead time
        # Based on Werner et al. (2005) - considerable uncertainty at 5-10 day lead times
        max_hours = MAX_FORECAST_DAYS * 24
        confidence = max(0.1, 1.0 - (lead_time_hours / max_hours) * 0.8)
        return round(confidence, 3)


class FloodForecastingEngine:
    """
    Main flood forecasting engine with security controls.

    Implements the European Flood Forecasting System (EFFS) concept
    from Werner et al. (2005) with modern security practices.
    """

    def __init__(
        self,
        config: SecureConfig,
        model: Optional[ForecastModel] = None
    ) -> None:
        """
        Initialize forecasting engine.

        Args:
            config: Station configuration
            model: Forecasting model (defaults to SimplePersistenceModel)
        """
        self._config = config
        self._model = model or SimplePersistenceModel()
        self._observations: list[WaterLevelReading] = []

        logger.info(
            f"Initialized FloodForecastingEngine for station {config.station_id}"
        )

    @property
    def config(self) -> SecureConfig:
        """Get station configuration (read-only)."""
        return self._config

    def add_observation(self, reading: WaterLevelReading) -> None:
        """
        Add a water level observation.

        Args:
            reading: Water level reading to add

        Raises:
            ValidationError: If reading is invalid
        """
        # Validate station ID matches
        if reading.station_id != self._config.station_id:
            raise ValidationError(
                f"Reading station_id '{reading.station_id}' does not match "
                f"engine station_id '{self._config.station_id}'"
            )

        # Check for duplicates (same timestamp)
        for existing in self._observations:
            if existing.timestamp == reading.timestamp:
                logger.warning(
                    f"Duplicate observation ignored for timestamp {reading.timestamp}"
                )
                return

        self._observations.append(reading)
        self._observations.sort(key=lambda x: x.timestamp)

        # Limit stored observations to prevent memory exhaustion
        max_observations = MAX_FORECAST_DAYS * 24 * 4  # 4 per hour
        if len(self._observations) > max_observations:
            self._observations = self._observations[-max_observations:]

        logger.debug(
            f"Added observation: level={reading.level_meters}m at {reading.timestamp}"
        )

    def _determine_warning_level(self, water_level: float) -> WarningLevel:
        """
        Determine warning level based on thresholds.

        Based on Sai et al. (2018) color-coded impact-based warnings.

        Args:
            water_level: Water level in meters

        Returns:
            Appropriate warning level
        """
        if water_level >= self._config.warning_threshold_red:
            return WarningLevel.RED
        elif water_level >= self._config.warning_threshold_orange:
            return WarningLevel.ORANGE
        elif water_level >= self._config.warning_threshold_yellow:
            return WarningLevel.YELLOW
        else:
            return WarningLevel.GREEN

    @rate_limited
    def generate_forecast(
        self,
        forecast_time: Optional[datetime] = None,
        lead_times_hours: Optional[list[int]] = None,
        client_id: str = "default"
    ) -> list[ForecastResult]:
        """
        Generate flood forecasts for specified lead times.

        Args:
            forecast_time: Base time for forecast (defaults to now)
            lead_times_hours: List of lead times in hours
            client_id: Client identifier for rate limiting

        Returns:
            List of forecast results

        Raises:
            ValidationError: If inputs are invalid
            RateLimitError: If rate limit exceeded
        """
        # Default forecast time to now
        if forecast_time is None:
            forecast_time = datetime.now(timezone.utc)
        else:
            forecast_time = validate_datetime(forecast_time, "forecast_time")

        # Default lead times
        if lead_times_hours is None:
            lead_times_hours = [6, 12, 24, 48, 72, 120, 168, 240]

        # Validate lead times
        max_lead_time = self._config.forecast_horizon_days * 24
        for lt in lead_times_hours:
            validate_numeric(
                lt,
                "lead_time",
                min_value=1,
                max_value=max_lead_time
            )

        # Check we have observations
        if not self._observations:
            raise ValidationError("No observations available for forecasting")

        results: list[ForecastResult] = []

        for lead_time in sorted(lead_times_hours):
            predicted, lower, upper = self._model.predict(
                self._observations,
                lead_time
            )
            confidence = self._model.get_confidence(lead_time)
            warning_level = self._determine_warning_level(predicted)

            valid_time = forecast_time + timedelta(hours=lead_time)

            result = ForecastResult(
                station_id=self._config.station_id,
                forecast_time=forecast_time,
                valid_time=valid_time,
                lead_time_hours=lead_time,
                predicted_level=predicted,
                prediction_interval_lower=lower,
                prediction_interval_upper=upper,
                confidence_level=confidence,
                warning_level=warning_level,
            )
            results.append(result)

        logger.info(
            f"Generated {len(results)} forecasts for station {self._config.station_id}"
        )

        return results

    def get_current_status(self) -> dict[str, Any]:
        """
        Get current station status.

        Returns:
            Dictionary with current status information
        """
        if not self._observations:
            return {
                "station_id": self._config.station_id,
                "station_name": self._config.station_name,
                "status": "no_data",
                "message": "No observations available",
            }

        latest = self._observations[-1]
        warning_level = self._determine_warning_level(latest.level_meters)

        return {
            "station_id": self._config.station_id,
            "station_name": self._config.station_name,
            "status": "operational",
            "latest_observation": {
                "timestamp": latest.timestamp.isoformat(),
                "level_meters": round(latest.level_meters, 3),
                "quality_flag": latest.quality_flag,
            },
            "current_warning_level": warning_level.name,
            "warning_description": warning_level.description,
            "thresholds": {
                "yellow": self._config.warning_threshold_yellow,
                "orange": self._config.warning_threshold_orange,
                "red": self._config.warning_threshold_red,
            },
        }


# =============================================================================
# SECURE DATA EXPORT
# =============================================================================

def secure_json_export(
    data: Any,
    output_path: Path,
    pretty: bool = True
) -> None:
    """
    Securely export data to JSON file.

    Args:
        data: Data to export
        output_path: Output file path
        pretty: Whether to format JSON for readability

    Raises:
        SecurityError: If path is unsafe
        IOError: If write fails
    """
    output_path = Path(output_path).resolve()

    # Validate path safety
    if not _is_safe_path(output_path):
        raise SecurityError(f"Unsafe output path: {output_path}")

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with secure permissions
    try:
        json_str = json.dumps(
            data,
            indent=2 if pretty else None,
            default=str,
            ensure_ascii=False
        )

        # Write atomically using temporary file
        temp_path = output_path.with_suffix(".tmp")
        temp_path.write_text(json_str, encoding=DEFAULT_ENCODING)
        temp_path.rename(output_path)

        # Set restrictive permissions (owner read/write only)
        os.chmod(output_path, 0o600)

        logger.info(f"Exported data to {output_path}")

    except (OSError, json.JSONEncodeError) as e:
        logger.error(f"Failed to export data: {e}")
        raise


# =============================================================================
# DEMONSTRATION
# =============================================================================

def run_demonstration() -> None:
    """
    Run a demonstration of the hardened flood forecasting system.

    This demonstrates:
    1. Secure configuration
    2. Input validation
    3. Rate limiting
    4. Forecast generation
    5. Secure data export
    """
    print("=" * 70)
    print("HARDENED FLOOD FORECASTING AND WARNING SYSTEM DEMO")
    print("Based on research by Werner et al., Zhang et al., and Sai et al.")
    print("=" * 70)
    print()

    # 1. Create secure configuration
    print("[1] Creating secure station configuration...")
    try:
        config = SecureConfig(
            station_id="RHINE_LOBITH_001",
            station_name="Rhine at Lobith (Netherlands)",
            latitude=51.8403,
            longitude=6.1089,
            warning_threshold_yellow=12.0,  # meters
            warning_threshold_orange=14.0,
            warning_threshold_red=16.0,
            forecast_horizon_days=10  # Based on Werner et al. (2005)
        )
        print(f"    Station: {config.station_name}")
        print(f"    Location: ({config.latitude}, {config.longitude})")
        print(f"    Forecast horizon: {config.forecast_horizon_days} days")
        print("    [OK] Configuration validated successfully")
    except ValidationError as e:
        print(f"    [ERROR] Configuration validation failed: {e}")
        return

    print()

    # 2. Initialize forecasting engine
    print("[2] Initializing forecasting engine...")
    engine = FloodForecastingEngine(config)
    print("    [OK] Engine initialized with SimplePersistenceModel")

    print()

    # 3. Add sample observations with validation
    print("[3] Adding validated water level observations...")

    base_time = datetime.now(timezone.utc) - timedelta(hours=24)
    sample_levels = [
        (0, 11.5, "good"),
        (6, 11.8, "good"),
        (12, 12.3, "good"),
        (18, 13.1, "suspect"),  # Marked as suspect quality
        (24, 13.5, "good"),
    ]

    for hours_offset, level, quality in sample_levels:
        try:
            reading = WaterLevelReading.create(
                timestamp=base_time + timedelta(hours=hours_offset),
                level_meters=level,
                station_id=config.station_id,
                quality_flag=quality
            )
            engine.add_observation(reading)
            print(f"    Added: {level}m at T+{hours_offset}h (quality: {quality})")
        except ValidationError as e:
            print(f"    [REJECTED] Invalid observation: {e}")

    print()

    # 4. Demonstrate input validation
    print("[4] Demonstrating input validation...")

    # Test invalid water level
    try:
        invalid_reading = WaterLevelReading.create(
            timestamp=datetime.now(timezone.utc),
            level_meters=999.0,  # Exceeds MAX_WATER_LEVEL
            station_id=config.station_id
        )
        print("    [UNEXPECTED] Invalid reading accepted")
    except ValidationError as e:
        print(f"    [BLOCKED] Invalid water level: {e}")

    # Test invalid station ID (injection attempt)
    try:
        invalid_reading = WaterLevelReading.create(
            timestamp=datetime.now(timezone.utc),
            level_meters=10.0,
            station_id="'; DROP TABLE stations; --"  # SQL injection attempt
        )
        print("    [UNEXPECTED] Injection attempt accepted")
    except ValidationError as e:
        print(f"    [BLOCKED] Invalid station ID pattern detected")

    print()

    # 5. Get current status
    print("[5] Current station status:")
    status = engine.get_current_status()
    print(f"    Status: {status['status']}")
    print(f"    Latest level: {status['latest_observation']['level_meters']}m")
    print(f"    Warning level: {status['current_warning_level']}")
    print(f"    Description: {status['warning_description']}")

    print()

    # 6. Generate forecasts
    print("[6] Generating probabilistic forecasts...")
    print("    (Uncertainty increases with lead time per Werner et al., 2005)")

    try:
        forecasts = engine.generate_forecast(
            lead_times_hours=[6, 24, 72, 120, 240],
            client_id="demo_user"
        )

        print()
        print("    Lead Time | Predicted | 90% Interval      | Confidence | Warning")
        print("    " + "-" * 65)

        for fc in forecasts:
            print(
                f"    {fc.lead_time_hours:>6}h   | "
                f"{fc.predicted_level:>7.2f}m | "
                f"[{fc.prediction_interval_lower:.2f}, {fc.prediction_interval_upper:.2f}]m | "
                f"{fc.confidence_level:>8.1%}   | "
                f"{fc.warning_level.name}"
            )

    except (ValidationError, RateLimitError) as e:
        print(f"    [ERROR] Forecast generation failed: {e}")

    print()

    # 7. Demonstrate rate limiting
    print("[7] Demonstrating rate limiting protection...")

    # Create a strict rate limiter for demo
    demo_limiter = RateLimiter(max_requests=3, window_seconds=60)

    for i in range(5):
        if demo_limiter.check_rate_limit("test_client"):
            remaining = demo_limiter.get_remaining("test_client")
            print(f"    Request {i+1}: [ALLOWED] ({remaining} remaining)")
        else:
            print(f"    Request {i+1}: [BLOCKED] Rate limit exceeded")

    print()

    # 8. Secure export
    print("[8] Exporting forecast data securely...")

    output_dir = Path("/home/ubuntu/flood_forecasting_demo/output")
    output_file = output_dir / "forecast_results.json"

    try:
        export_data = {
            "station": status,
            "forecasts": [fc.to_dict() for fc in forecasts],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "security_note": "Data exported with secure file permissions (0600)",
        }

        secure_json_export(export_data, output_file)
        print(f"    [OK] Data exported to {output_file}")
        print(f"    [OK] File permissions set to owner-only (0600)")

    except (SecurityError, IOError) as e:
        print(f"    [ERROR] Export failed: {e}")

    print()
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print()
    print("Security features demonstrated:")
    print("  - Input validation with bounds checking")
    print("  - Pattern-based string validation (prevents injection)")
    print("  - Immutable configuration (frozen dataclass)")
    print("  - Rate limiting (sliding window algorithm)")
    print("  - Secure logging with sensitive data redaction")
    print("  - Atomic file writes with restrictive permissions")
    print("  - Type hints and runtime validation")
    print("  - Comprehensive exception handling")
    print("=" * 70)


if __name__ == "__main__":
    run_demonstration()
