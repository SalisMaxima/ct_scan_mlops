"""ONNX Inference wrapper for DualPathway model."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


class ONNXPredictor:
    """Wrapper for running inference with ONNX Runtime."""

    def __init__(self, model_path: str | Path, providers: list[str] | None = None):
        """Initialize ONNX session.

        Args:
            model_path: Path to .onnx model file.
            providers: List of execution providers (e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider']).
                      Defaults to CPU if None.
        """
        self.model_path = str(model_path)
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.model_path, providers=providers)

        # Get input names
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

    def predict(self, image: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            image: Preprocessed image array (Batch, 3, 224, 224).
            features: Normalized features array (Batch, FeatureDim).

        Returns:
            Logits array (Batch, NumClasses).
        """
        # Ensure inputs are float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if features.dtype != np.float32:
            features = features.astype(np.float32)

        inputs = {
            self.input_names[0]: image,
            self.input_names[1]: features,
        }

        return self.session.run(self.output_names, inputs)[0]

    def predict_proba(self, image: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Run inference and return probabilities."""
        logits = self.predict(image, features)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
