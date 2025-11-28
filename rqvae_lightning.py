import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from residual_quantized_vae import ResidualQuantizer
from models import Movie, EMBEDDING_MODEL_NAME # Import the constant

# --- New, Modular Component ---
class EmbeddingGenerator:
    """
    A dedicated class to handle the generation of embeddings from a text dataset.
    """
    def __init__(self, model_name=EMBEDDING_MODEL_NAME): # Use the imported constant
        self.model_name = model_name
        self.embedding_model = None

    def _load_model(self):
        """Lazy-loads the sentence transformer model."""
        if self.embedding_model is None:
            print(f"Loading embedding model: '{self.model_name}'...")
            self.embedding_model = SentenceTransformer(self.model_name)

    def generate_from_hub(self, repo_id: str) -> torch.Tensor:
        """
        Loads a dataset from the Hub, creates embedding strings, and generates embeddings.
        """
        self._load_model()
        
        print(f"Loading source data from '{repo_id}' to generate embeddings...")
        hub_dataset = load_dataset(repo_id, split="train")
        
        texts_to_embed = []
        for item in hub_dataset:
            if item.get('plot_summary'):
                movie = Movie(
                    movie_id=item.get('movie_id'),
                    title=item.get('title'),
                    genres=item.get('genres', []),
                    plot_summary=item.get('plot_summary', ''),
                    director=item.get('director', ''),
                    stars=item.get('stars', [])
                )
                texts_to_embed.append(movie.to_embedding_string())

        if not texts_to_embed:
            raise ValueError("No valid texts for embedding found in the dataset.")
            
        print(f"Generating embeddings for {len(texts_to_embed)} movies...")
        embeddings = self.embedding_model.encode(texts_to_embed, show_progress_bar=True)
        
        return torch.tensor(embeddings, dtype=torch.float32)


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
    A PyTorch Lightning DataModule that uses an EmbeddingGenerator to prepare data.
    """
    def __init__(self, enriched_repo_id: str, batch_size: int, num_workers: int = 4):
        super().__init__()
        self.repo_id = enriched_repo_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.embedding_generator = EmbeddingGenerator() # Composition
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # Download the source dataset.
        load_dataset(self.repo_id)

    def setup(self, stage: str = None):
        # Delegate embedding generation to the specialized class
        full_embeddings_tensor = self.embedding_generator.generate_from_hub(self.repo_id)
        self.full_dataset = TensorDataset(full_embeddings_tensor)
        
        print(f"Generated {len(self.full_dataset)} embeddings.")
        
        # Split the data
        train_size = int(0.8 * len(self.full_dataset))
        val_size = len(self.full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.full_dataset, [train_size, val_size])
        print(f"Data split into {len(self.train_dataset)} training and {len(self.val_dataset)} validation samples.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
