# Give me end 2 end pytorch code to explore semantic ids

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. Vector Quantization (VQ) Layer ---
# This layer maps a continuous vector to the closest discrete codebook vector.
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        # M: Number of codes (e.g., 1024)
        self.num_embeddings = num_embeddings
        # D: Dimension of the vector
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook: The learnable vectors (centroids)
        # Shape: (M, D)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Initialize codebook vectors uniformly
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        # Input 'z' is the vector to be quantized (the residual or initial embedding)
        # Shape: (Batch_Size, D)

        # --- Find the closest codebook vector ---
        # 1. Calculate distances using torch.cdist for efficiency and clarity.
        #    torch.cdist computes the Euclidean distance (p=2) between each vector in z
        #    and each vector in the codebook.
        #    dists shape: (Batch_Size, M)
        dists = torch.cdist(z, self.embedding.weight)

        # 2. Find the index of the closest codebook vector (the Semantic Token Index)
        #    encoding_indices shape: (Batch_Size)
        encoding_indices = torch.argmin(dists, dim=1)

        # 3. Get the quantized vector by direct lookup.
        #    This is far more efficient than creating a one-hot matrix and using matmul.
        #    z_q shape: (Batch_Size, D)
        z_q = self.embedding(encoding_indices)

        # --- VQ-VAE Loss Calculation (Used for training the RQ-VAE) ---
        # 1. Commitment Loss (Encourages z to commit to a code): ||z - z_q.detach()||^2
        commitment_loss = F.mse_loss(z_q.detach(), z) * self.commitment_cost
        # 2. Codebook Loss (Encourages codebook vectors to follow z): ||z.detach() - z_q||^2
        codebook_loss = F.mse_loss(z.detach(), z_q)

        # Stop gradient for z_q to apply VQ-VAE loss trick (pass z_q but use z's gradient)
        # The gradients of the VQ layer are passed back through the straight-through estimator (z_q - z.detach() + z)
        z_q = z + (z_q - z).detach()

        return z_q, encoding_indices, commitment_loss, codebook_loss


# --- 2. Residual Quantization (RQ) Module ---
# This module applies VQ sequentially to the residuals.
class ResidualQuantizer(nn.Module):
    def __init__(self, num_layers, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        # L: Number of quantization steps (SID length)
        self.num_layers = num_layers

        # Create a list of VQ layers (codebooks)
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            for _ in range(num_layers)
        ])

    def forward(self, z):
        # Input 'z': The initial dense item embedding
        # Shape: (Batch_Size, D)

        # Store results for each layer
        quantized_vectors = []
        indices = []
        total_commitment_loss = 0
        total_codebook_loss = 0

        # Initial residual is the input embedding
        residual = z.clone()

        # Iteratively quantize the residual
        for quantizer in self.quantizers:
            # 1. Quantize the current residual
            z_q_k, indices_k, c_loss_k, cb_loss_k = quantizer(residual)

            # 2. Update the residual for the next step: r_k = r_{k-1} - z_q_k
            # NOTE: We use the *actual* z_q_k here (before the straight-through estimator)
            # to calculate the true error for the next layer.
            residual = residual - z_q_k.detach()

            # Store results
            quantized_vectors.append(z_q_k)
            indices.append(indices_k)
            total_commitment_loss += c_loss_k
            total_codebook_loss += cb_loss_k

        # The final quantized vector is the sum of all quantized components
        # This is the best discrete approximation of the original embedding 'z'
        z_q_final = sum(quantized_vectors)

        # The Semantic ID is the sequence of indices (L tokens)
        # Shape: (L, Batch_Size) -> transpose to (Batch_Size, L)
        sid_tokens = torch.stack(indices, dim=0).t()

        # Total Loss is used to train the RQ-VAE
        rq_loss = total_commitment_loss + total_codebook_loss

        return z_q_final, sid_tokens, rq_loss

if __name__ == '__main__':
    # --- 3. Example Usage: Generating a Semantic ID ---
    # Hyperparameters
    EMBEDDING_DIM = 64  # D: Dimension of the item embedding
    NUM_LAYERS = 3  # L: Length of the Semantic ID (e.g., 3 tokens)
    NUM_EMBEDDINGS = 512  # M: Size of each codebook (e.g., 512 codes per layer)

    # Instantiate the RQ module
    rq_module = ResidualQuantizer(NUM_LAYERS, NUM_EMBEDDINGS, EMBEDDING_DIM)

    # Simulate an input item embedding (e.g., from a pre-trained content encoder)
    # Batch size of 4 items
    input_embedding = torch.randn(4, EMBEDDING_DIM)

    # Forward pass to generate SIDs
    z_q_approx, semantic_ids, rq_loss = rq_module(input_embedding)

    print(f"✅ Input Embedding Shape: {input_embedding.shape}")
    print(f"✅ Final Quantized Vector Shape: {z_q_approx.shape}")
    print(f"✅ Semantic IDs (Batch_Size x SID_Length): \n{semantic_ids}")
    print(f"✅ Example Semantic ID (Tokens): {semantic_ids[0].tolist()}")
    print(f"✅ RQ-VAE Loss (for training): {rq_loss.item():.4f}")