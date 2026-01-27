"""
Tests for robust checkpoint loading, specifically targeting issues with
`torch.load`'s `weights_only=True` default in PyTorch >= 2.6.
"""

from __future__ import annotations

import collections
import typing
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, listconfig, nodes
from omegaconf.base import ContainerMetadata, Metadata
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset


class SimpleTestModule(pl.LightningModule):
    """A minimal LightningModule that saves hyperparameters."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.layer = torch.nn.Linear(32, 4)
        # This is the line that causes issues by saving non-tensor data
        self.save_hyperparameters(cfg)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return torch.nn.functional.cross_entropy(y_hat, y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def test_checkpoint_loading_with_safe_globals(tmp_path: Path):
    """
    Emulates the UnpicklingError and verifies the `safe_globals` fix.

    This test:
    1. Creates a LightningModule with a complex `DictConfig` hyperparameter.
    2. Trains for one step to generate a checkpoint, which will contain
       the pickled `DictConfig` and its constituent types.
    3. Attempts to load this checkpoint for testing, which would fail
       under new PyTorch defaults.
    4. Wraps the loading call in `torch.serialization.safe_globals` to
       prove that allowlisting the unsafe classes resolves the error.
    """
    # 1. Create a complex config that includes types which are not
    #    considered safe by default in torch.load(weights_only=True)
    complex_cfg = OmegaConf.create(
        {
            "model": {"name": "test_model", "params": {"a": 1, "b": None}},
            "train": {
                "optimizer": "adam",
                "params": OmegaConf.create({"betas": [0.9, 0.999]}),
            },
            "some_list": [1, "foo", None],
        }
    )

    model = SimpleTestModule(complex_cfg)

    # 2. Create dummy data and a trainer to generate a checkpoint
    train_data = TensorDataset(torch.randn(10, 32), torch.randint(0, 4, (10,)))
    train_loader = DataLoader(train_data)
    test_loader = DataLoader(train_data)

    checkpoint_callback = ModelCheckpoint(dirpath=tmp_path, filename="test-checkpoint", save_top_k=1, monitor="step")

    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=1,
        limit_test_batches=1,
        callbacks=[checkpoint_callback],
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=train_loader)

    # 3. Get the path to the saved checkpoint
    ckpt_path = checkpoint_callback.best_model_path
    assert Path(ckpt_path).exists()

    # 4. Attempt to load the checkpoint for testing using the fix
    # This block will fail if the list of safe globals is incomplete.

    # These are the classes that have caused UnpicklingErrors so far.
    # By adding them to `safe_globals`, we explicitly trust them.
    unsafe_globals = [
        DictConfig,
        ContainerMetadata,
        Metadata,
        typing.Any,
        dict,
        collections.defaultdict,
        nodes.AnyNode,
        listconfig.ListConfig,
        list,
        int,
    ]

    with torch.serialization.safe_globals(unsafe_globals):
        # The .test() method internally calls torch.load()
        results = trainer.test(dataloaders=test_loader, ckpt_path=ckpt_path)

    assert "test_acc" in results[0]
    assert results[0]["test_acc"] >= 0.0
