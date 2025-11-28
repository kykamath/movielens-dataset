import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin # 1. Import the mixin

# --- 1. Vector Quantization (VQ) Layer ---
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        dists = torch.cdist(z, self.embedding.weight)
        encoding_indices = torch.argmin(dists, dim=1)
        z_q = self.embedding(encoding_indices)
        
        commitment_loss = F.mse_loss(z_q.detach(), z) * self.commitment_cost
        codebook_loss = F.mse_loss(z.detach(), z_q)
        
        z_q = z + (z_q - z).detach()
        return z_q, encoding_indices, commitment_loss, codebook_loss

# --- 2. Residual Quantization (RQ) Module ---
# 2. Inherit from nn.Module AND the mixin
class ResidualQuantizer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_layers, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        # We need to save the config for the mixin to work correctly
        self.config = {
            "num_layers": num_layers,
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "commitment_cost": commitment_cost,
        }
        self.num_layers = num_layers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            for _ in range(num_layers)
        ])

    def forward(self, z):
        quantized_vectors = []
        indices = []
        total_commitment_loss = 0
        total_codebook_loss = 0
        residual = z.clone()

        for quantizer in self.quantizers:
            z_q_k, indices_k, c_loss_k, cb_loss_k = quantizer(residual)
            residual = residual - z_q_k.detach()
            quantized_vectors.append(z_q_k)
            indices.append(indices_k)
            total_commitment_loss += c_loss_k
            total_codebook_loss += cb_loss_k

        z_q_final = sum(quantized_vectors)
        sid_tokens = torch.stack(indices, dim=0).t()
        rq_loss = total_commitment_loss + total_codebook_loss

        return z_q_final, sid_tokens, rq_loss

if __name__ == '__main__':
    # --- 3. Example Usage: Generating a Semantic ID ---
    EMBEDDING_DIM = 64
    NUM_LAYERS = 3
    NUM_EMBEDDINGS = 512

    rq_module = ResidualQuantizer(NUM_LAYERS, NUM_EMBEDDINGS, EMBEDDING_DIM)
    input_embedding = torch.randn(4, EMBEDDING_DIM)
    z_q_approx, semantic_ids, rq_loss = rq_module(input_embedding)

    print(f"✅ Input Embedding Shape: {input_embedding.shape}")
    print(f"✅ Final Quantized Vector Shape: {z_q_approx.shape}")
    print(f"✅ Semantic IDs (Batch_Size x SID_Length): \n{semantic_ids}")
    print(f"✅ Example Semantic ID (Tokens): {semantic_ids[0].tolist()}")
    print(f"✅ RQ-VAE Loss (for training): {rq_loss.item():.4f}")
