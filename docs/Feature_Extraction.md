# Feature Extraction Module Implementation Plan

## Overview

This document outlines the implementation plan for a hand-crafted feature extraction module designed to complement a CNN-based classification model for chest CT scan cancer classification. The module extracts radiomics-style features that have been scientifically validated to differentiate between:

- **Adenocarcinoma** (peripheral, ground-glass opacity patterns)
- **Squamous Cell Carcinoma** (central, cavitation-prone)
- **Large Cell Carcinoma** (peripheral, necrotic, aggressive)
- **Normal Lung Tissue** (homogeneous, low-density parenchyma)

---

## Scientific Background

### Why Hand-Crafted Features?

Radiomics features have been extensively validated in peer-reviewed literature for lung cancer subtype differentiation. Key findings:

1. **GLCM texture features** achieved AUC of 0.81 for differentiating adenocarcinoma from squamous cell carcinoma (Defined et al., 2020)
2. **Combined radiomics + deep learning** models outperform single-modality approaches, achieving up to 88% AUC (Machine learning studies, 2024)
3. **Squamous cell carcinoma shows intranodular homogeneity** while adenocarcinoma exhibits heterogeneity - directly measurable via texture features

### Radiological Characteristics by Cancer Type

| Cancer Type | Location | Key CT Features | Discriminating Characteristics |
|-------------|----------|-----------------|-------------------------------|
| Adenocarcinoma | Peripheral | Ground-glass opacity (GGO), air bronchograms, pleural retraction | Mixed solid/GGO, spiculated margins, heterogeneous texture |
| Squamous Cell | Central | Cavitation (up to 82%), bronchial obstruction | Homogeneous texture, well-defined margins, central necrosis |
| Large Cell | Peripheral | Large mass, focal necrosis, NO air bronchograms | Irregular margins, rapid growth pattern, heterogeneous |
| Normal | Throughout | Uniform parenchyma, 80% air/20% tissue | Homogeneous, low HU values (-700 to -900), thin vessel walls |

---

## Feature Categories

### Category 1: Intensity/Histogram Features (HIGH IMPACT)

**Scientific Rationale:**
- Hounsfield Unit (HU) statistics directly reflect tissue density
- Ground-glass opacity (adenocarcinoma) has lower HU than solid tumors (SCC)
- Normal lung parenchyma has characteristic low HU values (-700 to -900)
- Entropy measures tissue heterogeneity, elevated in malignant lesions

**Features to Extract:**

| Feature | Formula/Method | Clinical Relevance |
|---------|---------------|-------------------|
| `mean_intensity` | Mean pixel value | Overall tissue density |
| `std_intensity` | Standard deviation | Density variation |
| `skewness` | Third standardized moment | Asymmetry of intensity distribution |
| `kurtosis` | Fourth standardized moment | Presence of outlier intensities |
| `percentile_10` | 10th percentile | Low-density component (air/GGO) |
| `percentile_90` | 90th percentile | High-density component (solid tissue) |
| `intensity_range` | P90 - P10 | Dynamic range of densities |
| `entropy` | -sum(p * log(p)) | Heterogeneity/randomness |

**Implementation:**
```python
from scipy import stats
import numpy as np

def extract_intensity_features(image: np.ndarray) -> dict:
    """Extract intensity-based features from CT image."""
    flat = image.flatten()
    return {
        'mean_intensity': np.mean(flat),
        'std_intensity': np.std(flat),
        'skewness': stats.skew(flat),
        'kurtosis': stats.kurtosis(flat),
        'percentile_10': np.percentile(flat, 10),
        'percentile_90': np.percentile(flat, 90),
        'intensity_range': np.percentile(flat, 90) - np.percentile(flat, 10),
        'entropy': stats.entropy(np.histogram(flat, bins=256)[0] + 1e-10),
    }
```

---

### Category 2: GLCM Texture Features (HIGH IMPACT)

**Scientific Rationale:**
- Gray-Level Co-occurrence Matrix (GLCM) captures spatial relationships between pixels
- Studies show GLCM features (SRHGE, HGZE) achieve AUC of 0.81 for ADC vs SCC differentiation
- SCC tends toward intranodular homogeneity; adenocarcinoma shows heterogeneity
- Haralick features are robust across imaging conditions

**Features to Extract:**

| Feature | Description | Clinical Relevance |
|---------|-------------|-------------------|
| `contrast` | Local intensity variation | High in heterogeneous tumors (ADC, LCC) |
| `homogeneity` | Closeness to diagonal | High in SCC, normal tissue |
| `energy` | Sum of squared elements | Uniformity measure; high in homogeneous tissue |
| `correlation` | Linear dependency | Texture regularity |
| `entropy` | Randomness measure | High in malignant, chaotic tissue |
| `dissimilarity` | Similar to contrast | Edge/boundary detection |
| `ASM` | Angular Second Moment | Orderliness of texture |
| `cluster_shade` | Skewness of GLCM | Asymmetry in co-occurrences |

**Implementation:**
```python
from skimage.feature import graycomatrix, graycoprops
import numpy as np

def extract_glcm_features(image: np.ndarray,
                          distances: list = [1, 2, 3],
                          angles: list = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> dict:
    """Extract GLCM texture features from CT image."""
    # Normalize to 0-255 range for GLCM
    image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    # Compute GLCM
    glcm = graycomatrix(image_uint8, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)

    features = {}
    properties = ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity', 'ASM']

    for prop in properties:
        values = graycoprops(glcm, prop)
        features[f'glcm_{prop}_mean'] = np.mean(values)
        features[f'glcm_{prop}_std'] = np.std(values)

    # Compute entropy separately
    glcm_normalized = glcm / (glcm.sum() + 1e-10)
    entropy = -np.sum(glcm_normalized * np.log2(glcm_normalized + 1e-10))
    features['glcm_entropy'] = entropy

    return features
```

---

### Category 3: Shape/Morphological Features (HIGH IMPACT)

**Scientific Rationale:**
- Sphericity correlates with malignancy (irregular shapes = higher malignancy)
- Spiculated margins are characteristic of adenocarcinoma
- Compactness helps identify infiltrative growth patterns
- Shape features combined with SUVmax achieved AUC of 0.92 in studies

**Features to Extract:**

| Feature | Formula | Clinical Relevance |
|---------|---------|-------------------|
| `sphericity` | (36*pi*V^2)^(1/3) / A | Roundness; low in malignant |
| `compactness` | P^2 / (4*pi*A) | Circularity measure |
| `eccentricity` | From fitted ellipse | Elongation |
| `solidity` | Area / ConvexHullArea | Fill ratio; low if irregular |
| `extent` | Area / BoundingBoxArea | Spread measure |
| `perimeter` | Boundary length | Edge complexity |
| `area` | Pixel count | Size measure |
| `major_axis_length` | Fitted ellipse major axis | Size in primary direction |
| `minor_axis_length` | Fitted ellipse minor axis | Size in secondary direction |

**Implementation:**
```python
from skimage import measure
from skimage.measure import regionprops
import numpy as np

def extract_shape_features(image: np.ndarray, threshold: float = None) -> dict:
    """Extract shape/morphological features from CT image."""
    # Binarize image (segment region of interest)
    if threshold is None:
        threshold = np.mean(image)

    binary = image > threshold

    # Label connected components
    labeled = measure.label(binary)

    if labeled.max() == 0:
        # No regions found, return zeros
        return {k: 0.0 for k in ['area', 'perimeter', 'eccentricity', 'solidity',
                                  'extent', 'compactness', 'sphericity']}

    # Get largest region
    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)

    area = largest.area
    perimeter = largest.perimeter

    # Compute derived features
    compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
    sphericity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    return {
        'area': area,
        'perimeter': perimeter,
        'eccentricity': largest.eccentricity,
        'solidity': largest.solidity,
        'extent': largest.extent,
        'major_axis_length': largest.major_axis_length,
        'minor_axis_length': largest.minor_axis_length,
        'compactness': compactness,
        'sphericity': sphericity,
    }
```

---

### Category 4: Region-Based Features (MEDIUM IMPACT)

**Scientific Rationale:**
- Ground-glass opacity ratio is critical for adenocarcinoma subtyping
- Cavitation is present in up to 82% of squamous cell carcinomas
- Solid component ratio correlates with invasiveness in adenocarcinoma
- Low-density central regions indicate necrosis (LCC, advanced SCC)

**Features to Extract:**

| Feature | Method | Clinical Relevance |
|---------|--------|-------------------|
| `solid_ratio` | High-intensity pixels / total | Solid vs GGO balance |
| `ggo_ratio` | Mid-intensity pixels / total | Ground-glass component |
| `low_density_ratio` | Low-intensity central pixels | Cavitation/necrosis |
| `intensity_gradient` | Edge gradient magnitude | Margin sharpness |
| `boundary_irregularity` | Std of radial distances | Spiculation proxy |

**Implementation:**
```python
import numpy as np
from scipy import ndimage

def extract_region_features(image: np.ndarray) -> dict:
    """Extract region-based features for GGO/solid/cavity detection."""
    # Normalize image to 0-1
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)

    # Define thresholds (these may need calibration based on your data)
    # For CT, typically: air < -950 HU, GGO: -750 to -300 HU, solid > -300 HU
    total_pixels = img_norm.size

    # Approximate thresholds for normalized images
    low_threshold = 0.2    # Air/cavity regions
    ggo_threshold = 0.5    # Ground-glass opacity
    solid_threshold = 0.7  # Solid tissue

    low_density_pixels = np.sum(img_norm < low_threshold)
    ggo_pixels = np.sum((img_norm >= low_threshold) & (img_norm < ggo_threshold))
    solid_pixels = np.sum(img_norm >= solid_threshold)

    # Compute gradient for margin analysis
    gradient_x = ndimage.sobel(img_norm, axis=0)
    gradient_y = ndimage.sobel(img_norm, axis=1)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Boundary irregularity (standard deviation of edge gradients)
    edge_mask = gradient_magnitude > np.percentile(gradient_magnitude, 90)
    boundary_irregularity = np.std(gradient_magnitude[edge_mask]) if edge_mask.any() else 0

    return {
        'solid_ratio': solid_pixels / total_pixels,
        'ggo_ratio': ggo_pixels / total_pixels,
        'low_density_ratio': low_density_pixels / total_pixels,
        'mean_gradient': np.mean(gradient_magnitude),
        'max_gradient': np.max(gradient_magnitude),
        'boundary_irregularity': boundary_irregularity,
    }
```

---

### Category 5: Wavelet Features (MEDIUM IMPACT)

**Scientific Rationale:**
- Multi-resolution analysis captures texture at different scales
- Wavelet coefficients encode frequency information lost in spatial analysis
- Effective for detecting subtle tissue patterns in medical imaging

**Features to Extract:**

| Feature | Method | Clinical Relevance |
|---------|--------|-------------------|
| `wavelet_energy_LH` | Energy in LH subband | Horizontal edges |
| `wavelet_energy_HL` | Energy in HL subband | Vertical edges |
| `wavelet_energy_HH` | Energy in HH subband | Diagonal features |
| `wavelet_entropy` | Entropy across subbands | Complexity measure |

**Implementation:**
```python
import pywt
import numpy as np

def extract_wavelet_features(image: np.ndarray, wavelet: str = 'db4', level: int = 2) -> dict:
    """Extract wavelet-based texture features."""
    features = {}

    # Perform 2D wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    for i, detail_coeffs in enumerate(coeffs[1:], 1):
        LH, HL, HH = detail_coeffs

        # Energy in each subband
        features[f'wavelet_L{i}_LH_energy'] = np.sum(LH ** 2)
        features[f'wavelet_L{i}_HL_energy'] = np.sum(HL ** 2)
        features[f'wavelet_L{i}_HH_energy'] = np.sum(HH ** 2)

        # Mean absolute values
        features[f'wavelet_L{i}_LH_mean'] = np.mean(np.abs(LH))
        features[f'wavelet_L{i}_HL_mean'] = np.mean(np.abs(HL))
        features[f'wavelet_L{i}_HH_mean'] = np.mean(np.abs(HH))

    # Approximation coefficients
    approx = coeffs[0]
    features['wavelet_approx_energy'] = np.sum(approx ** 2)
    features['wavelet_approx_mean'] = np.mean(approx)

    return features
```

---

## Module Architecture

### File Structure

```
src/ct_scan_mlops/
├── features/
│   ├── __init__.py
│   ├── intensity.py      # Intensity/histogram features
│   ├── texture.py        # GLCM features
│   ├── shape.py          # Morphological features
│   ├── region.py         # Region-based features (GGO, cavity)
│   ├── wavelet.py        # Wavelet features
│   └── extractor.py      # Main FeatureExtractor class
├── model.py              # Updated with DualPathwayModel
└── data.py               # Updated dataset to return features
```

### Main Extractor Class

```python
# src/ct_scan_mlops/features/extractor.py

from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .intensity import extract_intensity_features
from .texture import extract_glcm_features
from .shape import extract_shape_features
from .region import extract_region_features
from .wavelet import extract_wavelet_features


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    use_intensity: bool = True
    use_glcm: bool = True
    use_shape: bool = True
    use_region: bool = True
    use_wavelet: bool = True
    glcm_distances: list = (1, 2, 3)
    wavelet_type: str = 'db4'
    wavelet_level: int = 2


class FeatureExtractor:
    """Extract hand-crafted radiomics features from CT images."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._feature_dim = None

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract all configured features from a single image.

        Args:
            image: 2D numpy array (H, W) or 3D (C, H, W)

        Returns:
            1D numpy array of features
        """
        # Handle channel dimension
        if image.ndim == 3:
            # Use grayscale (average channels or first channel)
            image = np.mean(image, axis=0) if image.shape[0] == 3 else image[0]

        features = {}

        if self.config.use_intensity:
            features.update(extract_intensity_features(image))

        if self.config.use_glcm:
            features.update(extract_glcm_features(image,
                                                   distances=list(self.config.glcm_distances)))

        if self.config.use_shape:
            features.update(extract_shape_features(image))

        if self.config.use_region:
            features.update(extract_region_features(image))

        if self.config.use_wavelet:
            features.update(extract_wavelet_features(image,
                                                      wavelet=self.config.wavelet_type,
                                                      level=self.config.wavelet_level))

        # Convert to array
        feature_array = np.array(list(features.values()), dtype=np.float32)

        # Handle NaN/Inf
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_array

    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of images.

        Args:
            images: Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, num_features)
        """
        batch_features = []
        images_np = images.cpu().numpy()

        for img in images_np:
            features = self.extract(img)
            batch_features.append(features)

        return torch.tensor(np.stack(batch_features), dtype=torch.float32)

    @property
    def feature_dim(self) -> int:
        """Get the dimension of the feature vector."""
        if self._feature_dim is None:
            # Extract from dummy image to determine dimension
            dummy = np.random.rand(224, 224).astype(np.float32)
            self._feature_dim = len(self.extract(dummy))
        return self._feature_dim

    def get_feature_names(self) -> list[str]:
        """Get names of all features being extracted."""
        dummy = np.random.rand(224, 224).astype(np.float32)
        features = {}

        if self.config.use_intensity:
            features.update(extract_intensity_features(dummy))
        if self.config.use_glcm:
            features.update(extract_glcm_features(dummy))
        if self.config.use_shape:
            features.update(extract_shape_features(dummy))
        if self.config.use_region:
            features.update(extract_region_features(dummy))
        if self.config.use_wavelet:
            features.update(extract_wavelet_features(dummy))

        return list(features.keys())
```

---

## Dual-Pathway Model Architecture

```python
# Addition to src/ct_scan_mlops/model.py

class DualPathwayModel(nn.Module):
    """Dual-pathway model combining CNN features with hand-crafted radiomics features.

    Architecture:
        Image Input
            |
            +---> CNN Backbone (ResNet18) ---> CNN Features (512-d)
            |                                        |
            +---> Feature Extractor ---> FC Layers ---> Radiomics Features (128-d)
                                                        |
                            Concatenate <---------------+
                                |
                            Fusion FC Layers
                                |
                            Classification
    """

    def __init__(
        self,
        num_classes: int = 4,
        feature_dim: int = 64,      # Dimension of radiomics feature branch output
        cnn_feature_dim: int = 512,  # Dimension of CNN feature output
        fusion_hidden: int = 256,    # Hidden dimension in fusion layers
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # CNN pathway (ResNet18 backbone)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.cnn_backbone = models.resnet18(weights=weights)

        # Remove the final FC layer, keep features
        cnn_in_features = self.cnn_backbone.fc.in_features
        self.cnn_backbone.fc = nn.Identity()

        # CNN feature projection
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_in_features, cnn_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Radiomics feature pathway (input will be hand-crafted features)
        # Feature dim will be set based on extractor output
        self.radiomics_projection = None  # Lazy initialization
        self.radiomics_input_dim = None
        self.feature_dim = feature_dim
        self.dropout = dropout

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(cnn_feature_dim + feature_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, num_classes),
        )

        # Optionally freeze CNN backbone
        if freeze_backbone:
            self.freeze_cnn_backbone()

    def _init_radiomics_projection(self, input_dim: int):
        """Lazily initialize the radiomics projection layer."""
        self.radiomics_input_dim = input_dim
        self.radiomics_projection = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.feature_dim),
            nn.ReLU(inplace=True),
        )
        # Move to same device as other parameters
        device = next(self.parameters()).device
        self.radiomics_projection = self.radiomics_projection.to(device)

    def freeze_cnn_backbone(self):
        """Freeze CNN backbone parameters."""
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False

    def unfreeze_cnn_backbone(self):
        """Unfreeze CNN backbone parameters."""
        for param in self.cnn_backbone.parameters():
            param.requires_grad = True

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through dual pathways.

        Args:
            image: Image tensor of shape (B, C, H, W)
            features: Hand-crafted features of shape (B, feature_dim)

        Returns:
            Classification logits of shape (B, num_classes)
        """
        # CNN pathway
        cnn_features = self.cnn_backbone(image)
        cnn_features = self.cnn_projection(cnn_features)

        # Radiomics pathway (lazy init)
        if self.radiomics_projection is None:
            self._init_radiomics_projection(features.shape[1])

        radiomics_features = self.radiomics_projection(features)

        # Fusion
        combined = torch.cat([cnn_features, radiomics_features], dim=1)
        output = self.fusion(combined)

        return output
```

---

## Implementation Steps

### Phase 1: Core Feature Extraction (Priority: HIGH)

1. **Create feature module structure**
   - Create `src/ct_scan_mlops/features/` directory
   - Implement `__init__.py` with exports

2. **Implement intensity features** (`intensity.py`)
   - Mean, std, skewness, kurtosis
   - Percentiles and entropy
   - Unit tests

3. **Implement GLCM features** (`texture.py`)
   - Contrast, homogeneity, energy, correlation
   - Multi-distance, multi-angle computation
   - Unit tests

4. **Implement shape features** (`shape.py`)
   - Sphericity, compactness, eccentricity
   - Solidity, extent
   - Unit tests

### Phase 2: Extended Features (Priority: MEDIUM)

5. **Implement region features** (`region.py`)
   - Solid/GGO/cavity ratios
   - Gradient-based margin analysis
   - Unit tests

6. **Implement wavelet features** (`wavelet.py`)
   - Multi-level decomposition
   - Subband energy and statistics
   - Unit tests

### Phase 3: Integration

7. **Create main extractor class** (`extractor.py`)
   - FeatureConfig dataclass
   - FeatureExtractor with batch processing
   - Feature normalization

8. **Update dataset classes** (`data.py`)
   - Modify `ChestCTDataset` to return (image, features, label)
   - Pre-compute features during preprocessing
   - Cache features to disk

9. **Implement DualPathwayModel** (`model.py`)
   - CNN backbone pathway
   - Radiomics feature pathway
   - Fusion layers

10. **Update training pipeline** (`train.py`)
    - Handle dual inputs
    - Log feature importance
    - Ablation study support

### Phase 4: Validation

11. **Feature analysis**
    - Correlation analysis between features
    - Feature importance ranking
    - Visualization of discriminative features

12. **Model evaluation**
    - Compare single-pathway vs dual-pathway
    - Per-class performance analysis
    - Confusion matrix analysis

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project.dependencies]
# Existing dependencies...
scikit-image = ">=0.21.0"  # For GLCM, regionprops
scipy = ">=1.11.0"         # For statistics
PyWavelets = ">=1.4.0"     # For wavelet features
```

---

## Configuration Integration

Add to Hydra config (`configs/model/dual_pathway.yaml`):

```yaml
model:
  name: dual_pathway
  num_classes: 4
  cnn_feature_dim: 512
  radiomics_feature_dim: 128
  fusion_hidden: 256
  dropout: 0.3
  pretrained: true
  freeze_backbone: false

features:
  use_intensity: true
  use_glcm: true
  use_shape: true
  use_region: true
  use_wavelet: true
  glcm_distances: [1, 2, 3]
  wavelet_type: "db4"
  wavelet_level: 2
```

---

## Expected Feature Dimensions

| Category | Features | Dimension |
|----------|----------|-----------|
| Intensity | 8 | 8 |
| GLCM | 6 properties x 2 (mean/std) + entropy | 13 |
| Shape | 9 | 9 |
| Region | 6 | 6 |
| Wavelet (2 levels) | 6 per level + 2 approx | 14 |
| **Total** | | **~50** |

---

## Testing Strategy

### Unit Tests

```python
# tests/test_features.py

import pytest
import numpy as np
from ct_scan_mlops.features import FeatureExtractor, FeatureConfig

def test_feature_extractor_output_shape():
    extractor = FeatureExtractor()
    image = np.random.rand(224, 224).astype(np.float32)
    features = extractor.extract(image)
    assert features.ndim == 1
    assert len(features) == extractor.feature_dim

def test_feature_extractor_batch():
    import torch
    extractor = FeatureExtractor()
    batch = torch.rand(4, 3, 224, 224)
    features = extractor.extract_batch(batch)
    assert features.shape == (4, extractor.feature_dim)

def test_features_no_nan():
    extractor = FeatureExtractor()
    # Test with edge cases
    edge_cases = [
        np.zeros((224, 224)),
        np.ones((224, 224)),
        np.random.rand(224, 224),
    ]
    for img in edge_cases:
        features = extractor.extract(img.astype(np.float32))
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
```

---

## References and Sources

### Primary Research Papers

1. **CT Radiomics for ADC vs SCC Differentiation**
   - Publication: European Journal of Radiology (2020)
   - URL: https://pubmed.ncbi.nlm.nih.gov/32361604/
   - Key Finding: GLCM features (SRHGE, HGZE) achieved AUC of 0.81; combined with shape_sphericity and SUVmax reached AUC of 0.92

2. **Radiomics Beyond Nodule Morphology (2024)**
   - Publication: Nature Scientific Reports
   - URL: https://www.nature.com/articles/s41598-024-83786-6
   - Key Finding: SCC shows intranodular homogeneity while adenocarcinoma exhibits heterogeneity

3. **Machine Learning for NSCLC Classification (2024)**
   - Publication: PubMed
   - URL: https://pubmed.ncbi.nlm.nih.gov/38568892/
   - Key Finding: MLP Classifier achieved 83% accuracy, 88% AUC using 101 radiomic features

4. **Deep Fusion GLCM for Lung Nodule Classification**
   - Publication: PLOS ONE (2022)
   - URL: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0274516
   - Key Finding: 3D-GLCM with LSTM achieved superior classification of benign/malignant/ambiguous nodules

5. **Fusion Algorithm for Lung Nodule Classification**
   - Publication: BMC Pulmonary Medicine (2023)
   - URL: https://bmcpulmmed.biomedcentral.com/articles/10.1186/s12890-023-02708-w
   - Key Finding: Combined texture (GLCM), shape (Fourier), and deep features at decision level

### Radiological Reference Sources

6. **Lung Adenocarcinoma CT Features**
   - Source: Radiopaedia
   - URL: https://radiopaedia.org/articles/adenocarcinoma-in-situ-minimally-invasive-adenocarcinoma-and-invasive-adenocarcinoma-of-lung-1
   - Key Points: GGO patterns, solid component significance, air bronchograms

7. **Revised Lung Adenocarcinoma Classification - Imaging Guide**
   - Publication: PMC (2014)
   - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC4209391/
   - Key Points: Spectrum from GGN to solid mass, solid component = invasive growth

8. **Squamous Cell Carcinoma of Lung**
   - Source: Radiopaedia
   - URL: https://radiopaedia.org/articles/squamous-cell-carcinoma-of-the-lung
   - Key Points: Central location, cavitation (up to 82%), bronchial obstruction

9. **Stage I Lung SCC CT Evolution (2025)**
   - Publication: Cancer Imaging
   - URL: https://link.springer.com/article/10.1186/s40644-025-00952-3
   - Key Points: Central vs peripheral types, imaging features, surgical prognosis

10. **Large Cell Lung Cancer**
    - Source: Radiopaedia
    - URL: https://radiopaedia.org/articles/large-cell-lung-cancer-1
    - Key Points: Large peripheral mass, focal necrosis, no air bronchograms, no calcification

11. **CT Imaging Patterns in Major Histological Types**
    - Publication: MDPI Life (2024)
    - URL: https://www.mdpi.com/2075-1729/14/4/462
    - Key Points: Location preferences, characteristic patterns by histology

### Normal Lung Reference

12. **Normal Lung CT Anatomy**
    - Source: Medmastery
    - URL: https://www.medmastery.com/magazine/how-identify-normal-lung-anatomy-chest-ct
    - Key Points: 80% air/20% tissue, uniform attenuation, thin bronchial walls

13. **Thin-Section CT of Lungs: The Hinterland of Normal**
    - Publication: Radiology (RSNA)
    - URL: https://pubs.rsna.org/doi/10.1148/radiol.10092307
    - Key Points: Normal variants, gravity-dependent changes, diagnostic challenges

### Technical Implementation References

14. **GLCM Feature Extraction**
    - Publication: Wiley Computational Methods in Medicine (2022)
    - URL: https://onlinelibrary.wiley.com/doi/10.1155/2022/2733965
    - Key Points: Ensemble learning with GLCM for early lung cancer detection

15. **PyRadiomics Documentation**
    - Source: PyRadiomics
    - URL: https://pyradiomics.readthedocs.io/
    - Key Points: Standard radiomics feature extraction library

---

## Appendix: Feature Extraction Mathematical Definitions

### GLCM Features

**Contrast:**
$$C = \sum_{i,j} (i-j)^2 P(i,j)$$

**Homogeneity (Inverse Difference Moment):**
$$H = \sum_{i,j} \frac{P(i,j)}{1 + (i-j)^2}$$

**Energy (Angular Second Moment):**
$$E = \sum_{i,j} P(i,j)^2$$

**Correlation:**
$$Corr = \sum_{i,j} \frac{(i - \mu_i)(j - \mu_j) P(i,j)}{\sigma_i \sigma_j}$$

**Entropy:**
$$S = -\sum_{i,j} P(i,j) \log(P(i,j))$$

### Shape Features

**Sphericity (2D approximation):**
$$Sphericity = \frac{4\pi \cdot Area}{Perimeter^2}$$

**Compactness:**
$$Compactness = \frac{Perimeter^2}{4\pi \cdot Area}$$

**Solidity:**
$$Solidity = \frac{Area}{ConvexHullArea}$$

---

*Document created: January 2025*
*Last updated: January 2025*
