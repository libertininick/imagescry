"""Pipelines for encapsulating model inference and storage."""

from tempfile import TemporaryDirectory
from typing import cast

from jaxtyping import jaxtyped
from lightning import LightningModule, Trainer
from lightning.pytorch.accelerators import Accelerator
from more_itertools import flatten
from torch.utils.data import DataLoader

from imagescry.data import EmbeddingBatch, ImageBatch
from imagescry.image.info import ImageInfos
from imagescry.models.decomposition import PCA
from imagescry.models.embedding import EmbeddingModule
from imagescry.storage.database import DatabaseManager
from imagescry.storage.models import Embedding
from imagescry.typechecking import typechecker


class EmbeddingPCAPipeline(LightningModule):
    """Pipeline that embeds images and then transforms the embeddings to lower-dimensional space using PCA."""

    def __init__(
        self,
        *,
        embedding_model: EmbeddingModule,
        pca: PCA,
        db_manager: DatabaseManager | None = None,
        image_infos: ImageInfos | None = None,
        pca_checkpoint_id: int | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            embedding_model (EmbeddingModule): Pretrained embedding model to use.
            pca (PCA): Pretrained PCA model to use.
            db_manager (DatabaseManager | None): Database manager for storing embeddings. Defaults to None.
            image_infos (ImageInfos | None): Image information for the embeddings. Defaults to None.
            pca_checkpoint_id (int | None): ID of the PCA checkpoint used for this embedding. Defaults to None.

        Raises:
            ValueError: If the PCA model is not fitted or if database manager is provided an either `image_infos` or
                `pca_checkpoint_id` is not provided.
        """
        super().__init__()

        if not pca.fitted:
            raise ValueError("PCA model must be fitted before it can be used in the pipeline.")

        if db_manager is not None and (image_infos is None or pca_checkpoint_id is None):
            raise ValueError(
                "If a database manager is provided, both `image_infos` and `pca_checkpoint_id` must be provided."
            )

        self.embedding_model = embedding_model
        self.pca = pca
        self.db_manager = db_manager
        self.image_infos = image_infos or ImageInfos(items=[])
        self.pca_checkpoint_id = pca_checkpoint_id

    @jaxtyped(typechecker=typechecker)
    def predict_step(self, batch: ImageBatch) -> EmbeddingBatch | list[int]:
        """Embed images and then transform the embeddings to lower-dimensional space using PCA.

        Args:
            batch (ImageBatch): Batch of images and their dataset indices.

        Returns:
            EmbeddingBatch | list[int]: Compressed embedding feature map for the batch of images, or list of
                ids of stored embeddings if `db_manager` is provided.
        """
        # Embed images
        batch_size = len(batch)
        full_embeddings = self.embedding_model.predict_step(batch)

        # Transform embedding vectors to lower-dimensional space
        compressed_flat_embeddings = self.pca.transform(full_embeddings.get_flat_vectors())

        # Reshape embeddings to original spatial dimensions
        compressed_embeddings = compressed_flat_embeddings.reshape(
            batch_size, *full_embeddings.spatial_dims, self.pca.num_components
        ).permute(0, 3, 1, 2)

        if self.db_manager is None:
            # If not storing in DB, return the compressed embeddings directly
            return EmbeddingBatch(indices=batch.indices, embeddings=compressed_embeddings)

        # Store embeddings in the database
        batch_image_infos = self.image_infos[batch.indices.cpu().tolist()]
        embeddings = [
            Embedding.create(image_info, compressed_embeddings[i].cpu(), checkpoint_id=self.pca_checkpoint_id)
            for i, image_info in enumerate(batch_image_infos)
        ]
        embedding_ids = self.db_manager.add_items(embeddings)
        return embedding_ids

    def predict(
        self,
        dataloader: DataLoader,
        *,
        accelerator: str | Accelerator = "auto",
        devices: list[int] | str | int = "auto",
    ) -> list[EmbeddingBatch] | list[int]:
        """Run pipeline prediction using the provided dataloader.

        Args:
            dataloader (DataLoader): Dataloader for the images to be processed.
            accelerator (str | Accelerator): Accelerator to use. Defaults to "auto".
            devices (list[int] | str | int): Devices to use. Defaults to "auto

        Returns:
            list[EmbeddingBatch] | list[int]: List of EmbeddingBatch objects if not using a database manager, or list of
                ids of stored embeddings if using a database manager.
        """
        # Run the prediction step using a Trainer
        with TemporaryDirectory() as temp_dir:
            trainer = Trainer(accelerator=accelerator, devices=devices, default_root_dir=temp_dir)
            predictions = trainer.predict(self, dataloader)

        # Return predictions
        if not predictions:
            return []
        elif self.db_manager is None:
            # Not storing in the database, return the list of EmbeddingBatch objects
            return cast(list[EmbeddingBatch], predictions)
        else:
            # Stored output in the database, return a list of ids
            flattened_predictions = list(flatten(predictions))
            return cast(list[int], flattened_predictions)
