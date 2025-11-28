import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
from residual_quantized_vae import ResidualQuantizer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger # 1. Import the logger
from datasets import load_dataset
from models import HUB_EMBEDDINGS_REPO_ID

# --- 1. Hyperparameters ---
EMBEDDING_DIM = 768
NUM_LAYERS = 4
NUM_EMBEDDINGS = 1024
COMMITMENT_COST = 0.25
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 500

# --- 2. The LightningModule (No changes needed here) ---
class RQVAE(pl.LightningModule):
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

# --- Main Execution Block ---
def main():
    # --- 3. Data Preparation ---
    print(f"Preparing Data from Hugging Face Hub: {HUB_EMBEDDINGS_REPO_ID}")
    hub_dataset = load_dataset(HUB_EMBEDDINGS_REPO_ID, split="train")
    embeddings_list = [item['all_mpnet_base_v2_embedding'] for item in hub_dataset if item['all_mpnet_base_v2_embedding']]
    if not embeddings_list:
        print("No embeddings found in the dataset. Exiting.")
        return
    full_embeddings_tensor = torch.tensor(embeddings_list, dtype=torch.float32)
    full_dataset = TensorDataset(full_embeddings_tensor)
    print(f"Loaded {len(full_dataset)} embeddings of dimension {full_embeddings_tensor.shape[1]}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)

    # --- 4. Training with Callbacks and Logger ---
    print("Initializing RQ-VAE model...")
    model = RQVAE(
        num_layers=NUM_LAYERS,
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        commitment_cost=COMMITMENT_COST,
        learning_rate=LEARNING_RATE
    )

    # --- Configure Callbacks and Logger ---
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='rqvae-best-model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )
    # 2. Instantiate the logger
    logger = TensorBoardLogger("tb_logs", name="rq_vae_model")

    print("Initializing PyTorch Lightning Trainer with callbacks and logger...")
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger  # 3. Pass the logger to the Trainer
    )

    print("Starting Training...")
    trainer.fit(model, train_loader, val_loader)

    print("\nTraining Complete.")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

    # --- 5. Semantic ID Generation (Inference) ---
    print("\nLoading best model from checkpoint to generate Semantic IDs...")
    best_model = RQVAE.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = best_model.to(device)

    all_semantic_ids = []
    with torch.no_grad():
        _, ids, _ = best_model(full_embeddings_tensor.to(device))
        all_semantic_ids = ids.cpu().numpy()

    print("\n--- Example Semantic IDs from Best Model ---")
    for i in range(10):
        sid = all_semantic_ids[i]
        sid_str = " ".join([f"<T{j+1}:{token:04d}>" for j, token in enumerate(sid)])
        print(f"Movie {i + 1} SID: {sid_str}")

if __name__ == '__main__':
    main()
