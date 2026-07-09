class ModelInferenceError(Exception):
    """Exception raised when model inference fails."""
    def __init__(self, message: str, payload: dict | None = None):
        self.message = message
        self.payload = payload
        super().__init__(self.message)

class DataValidationError(Exception):
    """Exception raised when input data validation fails."""
    pass

class ModelLoadError(Exception):
    """Exception raised when the model artifact cannot be loaded."""
    pass
