# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader, TensorDataset, random_split
# from residual_quantized_vae import ResidualQuantizer
# from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger
# from datasets import load_dataset
# from models import HUB_EMBEDDINGS_REPO_ID
# from huggingface_hub import login
# from dotenv import load_dotenv
# import os
#
# # --- 1. Hyperparameters ---
# EMBEDDING_DIM = 768
# NUM_LAYERS = 4
# NUM_EMBEDDINGS = 1024
# COMMITMENT_COST = 0.25
# LEARNING_RATE = 1e-4
# BATCH_SIZE = 128
# EPOCHS = 500 # Changed from 100 to 500
# HUB_MODEL_ID = "krishnakamath/rq-vae-movielens"
#
# # --- 2. The LightningModule ---
# class RQVAE(pl.LightningModule):
#     def __init__(self, num_layers, num_embeddings, embedding_dim, commitment_cost, learning_rate):
#         super().__init__()
#         self.save_hyperparameters()
#         self.quantizer = ResidualQuantizer(
#             num_layers=self.hparams.num_layers,
#             num_embeddings=self.hparams.num_embeddings,
#             embedding_dim=self.hparams.embedding_dim,
#             commitment_cost=self.hparams.commitment_cost
#         )
#
#     def forward(self, z):
#         return self.quantizer(z)
#
#     def _common_step(self, batch, batch_idx):
#         batch_z, = batch
#         z_quantized, _, vq_loss = self.quantizer(batch_z)
#         reconstruction_loss = F.mse_loss(z_quantized, batch_z)
#         total_loss = reconstruction_loss + vq_loss
#         return total_loss, reconstruction_loss, vq_loss
#
#     def training_step(self, batch, batch_idx):
#         total_loss, recon_loss, vq_loss = self._common_step(batch, batch_idx)
#         self.log('train_loss', total_loss, prog_bar=True)
#         self.log('train_recon_loss', recon_loss, on_epoch=True, on_step=False)
#         self.log('train_vq_loss', vq_loss, on_epoch=True, on_step=False)
#         return total_loss
#
#     def validation_step(self, batch, batch_idx):
#         total_loss, recon_loss, vq_loss = self._common_step(batch, batch_idx)
#         self.log('val_loss', total_loss, prog_bar=True)
#         self.log('val_recon_loss', recon_loss, on_epoch=True)
#         self.log('val_vq_loss', vq_loss, on_epoch=True)
#         return total_loss
#
#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
#
# # --- Main Execution Block ---
# def main():
#     # --- 3. Authentication and Data Preparation ---
#     load_dotenv()
#     hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
#     if hf_token:
#         print("Logging in to Hugging Face Hub...")
#         login(token=hf_token)
#     else:
#         print("Warning: HUGGING_FACE_HUB_TOKEN not found. Model will not be uploaded.")
#
#     print(f"Preparing Data from Hugging Face Hub: {HUB_EMBEDDINGS_REPO_ID}")
#     hub_dataset = load_dataset(HUB_EMBEDDINGS_REPO_ID, split="train")
#     embeddings_list = [item['all_mpnet_base_v2_embedding'] for item in hub_dataset if item['all_mpnet_base_v2_embedding']]
#     if not embeddings_list:
#         print("No embeddings found in the dataset. Exiting.")
#         return
#     full_embeddings_tensor = torch.tensor(embeddings_list, dtype=torch.float32)
#     full_dataset = TensorDataset(full_embeddings_tensor)
#     print(f"Loaded {len(full_dataset)} embeddings of dimension {full_embeddings_tensor.shape[1]}")
#
#     train_size = int(0.8 * len(full_dataset))
#     val_size = len(full_dataset) - train_size
#     train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
#
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
#
#     # --- 4. Training with Callbacks and Logger ---
#     print("Initializing RQ-VAE model...")
#     model = RQVAE(
#         num_layers=NUM_LAYERS,
#         num_embeddings=NUM_EMBEDDINGS,
#         embedding_dim=EMBEDDING_DIM,
#         commitment_cost=COMMITMENT_COST,
#         learning_rate=LEARNING_RATE
#     )
#
#     early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')
#     checkpoint_callback = ModelCheckpoint(
#         monitor='val_loss',
#         dirpath='checkpoints/',
#         filename='rqvae-best-model-{epoch:02d}-{val_loss:.4f}',
#         save_top_k=1,
#         mode='min',
#     )
#     logger = TensorBoardLogger("tb_logs", name="rq_vae_model")
#
#     print("Initializing PyTorch Lightning Trainer...")
#     trainer = pl.Trainer(
#         max_epochs=EPOCHS,
#         accelerator="auto",
#         callbacks=[early_stop_callback, checkpoint_callback],
#         logger=logger
#     )
#
#     print("Starting Training...")
#     trainer.fit(model, train_loader, val_loader)
#
#     print("\nTraining Complete.")
#     print(f"Best model saved locally at: {checkpoint_callback.best_model_path}")
#
#     # --- 5. Upload Best Model to Hugging Face Hub ---
#     if hf_token and checkpoint_callback.best_model_path:
#         print(f"\nUploading best model from '{checkpoint_callback.best_model_path}' to Hugging Face Hub...")
#
#         best_model = RQVAE.load_from_checkpoint(checkpoint_callback.best_model_path)
#         quantizer_model = best_model.quantizer
#
#         # The push_to_hub method now exists thanks to the PyTorchModelHubMixin
#         quantizer_model.push_to_hub(
#             repo_id=HUB_MODEL_ID,
#             commit_message=f"Upload best model from epoch {best_model.current_epoch} with val_loss {checkpoint_callback.best_model_score:.4f}"
#         )
#         print(f"✅ Model successfully uploaded to {HUB_MODEL_ID}")
#     else:
#         print("\nSkipping model upload to Hugging Face Hub.")
#
#     # --- 6. Semantic ID Generation (Inference) ---
#     print("\nLoading best model to generate Semantic IDs...")
#
#     if checkpoint_callback.best_model_path:
#         inference_model = RQVAE.load_from_checkpoint(checkpoint_callback.best_model_path)
#     else:
#         print("No local checkpoint found. This should not happen if training ran.")
#         return
#
#     inference_model.eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     inference_model = inference_model.to(device)
#
#     all_semantic_ids = []
#     with torch.no_grad():
#         _, ids, _ = inference_model(full_embeddings_tensor.to(device))
#         all_semantic_ids = ids.cpu().numpy()
#
#     print("\n--- Example Semantic IDs from Best Model ---")
#     for i in range(10):
#         sid = all_semantic_ids[i]
#         sid_str = " ".join([f"<T{j+1}:{token:04d}>" for j, token in enumerate(sid)])
#         print(f"Movie {i + 1} SID: {sid_str}")
#
# if __name__ == '__main__':
#     main()
