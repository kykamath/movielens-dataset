import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
from datasets import load_dataset
from residual_quantized_vae import ResidualQuantizer
from models import HUB_EMBEDDINGS_REPO_ID

class RQVAE(pl.LightningModule):
    """
    The PyTorch Lightning module that wraps the ResidualQuantizer for training.
    """
    def __init__(self, num_layers, num_embeddings, embedding_dim, commitment_cost, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.quantizer = ResidualQuantizer(
            num_layers=self.hparams.num_layers,
            num_embeddings=self.hparams.num_embeddings,
            embedding_dim=self.hparams.embedding_dim,
            commitment_cost=self.hparams.commitment_cost
        )

    def forward(self, z):
        return self.quantizer(z)

    def _common_step(self, batch, batch_idx):
        """A common step for both training and validation."""
        batch_z, = batch
        z_quantized, _, vq_loss = self.quantizer(batch_z)
        reconstruction_loss = F.mse_loss(z_quantized, batch_z)
        total_loss = reconstruction_loss + vq_loss
        return total_loss, reconstruction_loss, vq_loss

    def training_step(self, batch, batch_idx):
        total_loss, recon_loss, vq_loss = self._common_step(batch, batch_idx)
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_recon_loss', recon_loss, on_epoch=True, on_step=False)
        self.log('train_vq_loss', vq_loss, on_epoch=True, on_step=False)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, recon_loss, vq_loss = self._common_step(batch, batch_idx)
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_epoch=True)
        self.log('val_vq_loss', vq_loss, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class MovieEmbeddingDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling the movie embeddings dataset.
    """
    def __init__(self, repo_id: str, batch_size: int, num_workers: int = 4):
        super().__init__()
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # This method is called only once on one process.
        # Use it to download the dataset.
        load_dataset(self.repo_id)

    def setup(self, stage: str = None):
        # This method is called on every GPU process.
        # Use it to load, split, and process the data.
        hub_dataset = load_dataset(self.repo_id, split="train")
        embeddings_list = [item['all_mpnet_base_v2_embedding'] for item in hub_dataset if item['all_mpnet_base_v2_embedding']]
        
        if not embeddings_list:
            raise ValueError("No embeddings found in the dataset.")
            
        full_embeddings_tensor = torch.tensor(embeddings_list, dtype=torch.float32)
        self.full_dataset = TensorDataset(full_embeddings_tensor)
        
        print(f"Loaded {len(self.full_dataset)} embeddings.")
        
        # Split the data
        train_size = int(0.8 * len(self.full_dataset))
        val_size = len(self.full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
