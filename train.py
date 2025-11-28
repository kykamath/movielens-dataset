import os
from dotenv import load_dotenv
from huggingface_hub import login
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from rqvae_lightning import RQVAE, MovieEmbeddingDataModule
from models import HUB_ENRICHED_REPO_ID, HUB_MODEL_ID # Import HUB_MODEL_ID

# --- 1. Hyperparameters ---
# Model Hyperparameters
EMBEDDING_DIM = 768
NUM_LAYERS = 4
NUM_EMBEDDINGS = 1024
COMMITMENT_COST = 0.25

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 500
# HUB_MODEL_ID is now imported from models.py

def main():
    # --- 2. Authentication and Setup ---
    load_dotenv()
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("Logging in to Hugging Face Hub...")
        login(token=hf_token)
    else:
        print("Warning: HUGGING_FACE_HUB_TOKEN not found. Model will not be uploaded.")

    # --- 3. Initialize DataModule ---
    print("Initializing DataModule...")
    data_module = MovieEmbeddingDataModule(enriched_repo_id=HUB_ENRICHED_REPO_ID, batch_size=BATCH_SIZE)

    # --- 4. Initialize Model ---
    print("Initializing RQ-VAE model...")
    model = RQVAE(
        num_layers=NUM_LAYERS,
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        commitment_cost=COMMITMENT_COST,
        learning_rate=LEARNING_RATE
    )

    # --- 5. Configure Callbacks and Logger ---
    print("Configuring callbacks and logger...")
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='rqvae-best-model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )
    logger = TensorBoardLogger("tb_logs", name="rq_vae_model")

    # --- 6. Initialize and Run Trainer ---
    print("Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger
    )

    print("Starting Training...")
    trainer.fit(model, datamodule=data_module)

    print("\nTraining Complete.")
    print(f"Best model saved locally at: {checkpoint_callback.best_model_path}")

    # --- 7. Upload Best Model to Hugging Face Hub ---
    if hf_token and checkpoint_callback.best_model_path:
        print(f"\nUploading best model from '{checkpoint_callback.best_model_path}' to Hugging Face Hub...")
        best_model = RQVAE.load_from_checkpoint(checkpoint_callback.best_model_path)
        quantizer_model = best_model.quantizer
        
        quantizer_model.push_to_hub(
            repo_id=HUB_MODEL_ID,
            commit_message=f"Upload best model from epoch {best_model.current_epoch} with val_loss {checkpoint_callback.best_model_score:.4f}"
        )
        print(f"✅ Model successfully uploaded to {HUB_MODEL_ID}")
    else:
        print("\nSkipping model upload to Hugging Face Hub.")

if __name__ == '__main__':
    main()
