"""Storage utility functions."""

from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from lightning import __version__ as pl_version
from sqlmodel import String, TypeDecorator


class PathType(TypeDecorator):
    """Custom SQLAlchemy type for python Path objects."""

    impl = String
    cache_ok = True

    @staticmethod
    def process_bind_param(value: Path | None, _: Any) -> str | None:
        """Convert Path to string when storing."""
        if value is not None:
            return str(value)
        return value

    @staticmethod
    def process_result_value(value: str | None, _: Any) -> Path | None:
        """Convert string back to Path when loading."""
        if value is not None:
            return Path(value)
        return value


def create_lightning_checkpoint(model: LightningModule) -> bytes:
    """Create a Lightning checkpoint with no training state.

    Args:
        model (LightningModule): The Lightning model to create a checkpoint for.

    Returns:
        bytes: The checkpoint data as bytes.
    """
    # Define the checkpoint dictionary without any training state
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": model.hparams,
        "pytorch-lightning_version": pl_version,
        "epoch": 0,
        "global_step": 0,
        "loops": {},
        "optimizer_states": [],
        "lr_schedulers": [],
        "callbacks": {},
        "hparams_name": "hparams",  # Name of the hyperparameters attribute
    }

    # Save the checkpoint to buffer
    with BytesIO() as buffer:
        torch.save(checkpoint, buffer)

        # Ensure the buffer is at the beginning
        buffer.seek(0)

        # Return the buffer content as bytes
        return buffer.getvalue()
