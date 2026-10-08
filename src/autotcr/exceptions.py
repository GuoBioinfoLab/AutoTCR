"""Domain-specific exceptions with actionable user messages."""


class AutoTCRError(Exception):
    """Base exception for expected AutoTCR failures."""


class ConfigurationError(AutoTCRError):
    """Raised when inference configuration is invalid or incomplete."""


class InputFormatError(AutoTCRError):
    """Raised when an input table cannot be interpreted safely."""


class ModelLoadError(AutoTCRError):
    """Raised when model components or checkpoint weights cannot be loaded."""

