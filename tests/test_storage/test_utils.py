"""Tests for storage utility functions."""

from io import BytesIO

import pytest
import torch
from lightning import LightningModule
from pytest_check import check_functions

from imagescry.storage.utils import create_lightning_checkpoint


class LightningModel(LightningModule):
    """A simple Lightning model for testing purposes."""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        """Initialize the test Lightning model."""
        super().__init__()
        self.save_hyperparameters({
            "num_inputs": num_inputs,
            "num_outputs": num_outputs,
        })

        # Define a simple linear layer
        self.layer = torch.nn.Linear(num_inputs, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.layer(x)


@pytest.fixture(scope="session")
def lightning_model() -> LightningModel:
    """Fixture for creating a LightningModel instance."""
    return LightningModel(num_inputs=10, num_outputs=5)


def test_create_lightning_checkpoint(lightning_model: LightningModel) -> None:
    """Test create_lightning_checkpoint function creates a checkpoint that can be loaded using load_from_checkpoint."""
    # Create a checkpoint from the Lightning model
    checkpoint = create_lightning_checkpoint(lightning_model)

    # Load from checkpoint to ensure it can be loaded correctly
    loaded_model = LightningModel.load_from_checkpoint(checkpoint_path=BytesIO(checkpoint))

    # Check that the loaded model has the same hyperparameters
    check_functions.equal(loaded_model.hparams, lightning_model.hparams, "Hyperparameters do not match")

    # Check layer weights match
    check_functions.is_true(
        torch.allclose(loaded_model.layer.weight, lightning_model.layer.weight), "Layer weights do not match"
    )
