# Model Module

Neural network architectures for CT scan classification.

## Overview

The model module provides:

- **CustomCNN** - A configurable CNN baseline
- **ResNet18** - Transfer learning with pretrained weights
- **Model registry** - Extensible system for adding new models

## Registry

Models are registered using the `@register_model` decorator and can be instantiated via `build_model()`.

::: ct_scan_mlops.model.build_model

::: ct_scan_mlops.model.register_model

## Models

::: ct_scan_mlops.model.CustomCNN
    options:
      heading_level: 3
      members:
        - __init__
        - forward
        - from_config

::: ct_scan_mlops.model.ResNet18
    options:
      heading_level: 3
      members:
        - __init__
        - forward
        - from_config
        - freeze_backbone
        - unfreeze_backbone

## Usage Examples

### Build from Config

```python
from ct_scan_mlops.model import build_model

model = build_model(cfg)  # cfg.model.name determines which model
```

### Direct Instantiation

```python
from ct_scan_mlops.model import CustomCNN, ResNet18

# CustomCNN
cnn = CustomCNN(
    num_classes=4,
    hidden_dims=[32, 64, 128, 256],
    fc_hidden=512,
    dropout=0.3,
)

# ResNet18 with transfer learning
resnet = ResNet18(
    num_classes=4,
    pretrained=True,
    freeze_backbone=False,
)
```

### Fine-tuning ResNet18

```python
from ct_scan_mlops.model import ResNet18

# Start with frozen backbone
model = ResNet18(num_classes=4, freeze_backbone=True)

# Train classification head only
train(model, epochs=10)

# Unfreeze and fine-tune entire model
model.unfreeze_backbone()
train(model, epochs=20, lr=1e-5)
```

## Adding New Models

1. Define your model class with `from_config` classmethod
2. Register it with the `@register_model` decorator

```python
from ct_scan_mlops.model import register_model

@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # ... define layers ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... forward pass ...

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "MyModel":
        return cls(num_classes=cfg.model.num_classes)
```

Then create `configs/model/my_model.yaml`:

```yaml
name: my_model
num_classes: 4
# ... other params ...
```

Use it:

```bash
invoke train --args "model=my_model"
```
