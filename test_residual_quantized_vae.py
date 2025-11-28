import torch
import pytest
from residual_quantized_vae import VectorQuantizer, ResidualQuantizer

# --- Test Fixture for VectorQuantizer ---
# This sets up a reusable VectorQuantizer instance and sample data for tests.
@pytest.fixture
def vq_setup():
    """Provides a standard VectorQuantizer instance and input tensor for tests."""
    num_embeddings = 64  # M: Size of the codebook
    embedding_dim = 16   # D: Dimension of the vectors
    batch_size = 8
    
    # Instantiate the layer
    quantizer = VectorQuantizer(num_embeddings, embedding_dim)
    
    # Create a sample input tensor
    z = torch.randn(batch_size, embedding_dim)
    
    return {
        "quantizer": quantizer,
        "z": z,
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size
    }

# --- Tests for the VectorQuantizer Class ---

def test_vq_forward_pass_shapes(vq_setup):
    """
    Tests if the VQ forward pass produces tensors of the correct shape.
    """
    quantizer = vq_setup["quantizer"]
    z = vq_setup["z"]
    batch_size = vq_setup["batch_size"]
    embedding_dim = vq_setup["embedding_dim"]

    z_q, indices, _, _ = quantizer(z)

    # 1. Check the shape of the quantized output vector
    assert z_q.shape == (batch_size, embedding_dim), \
        f"Expected z_q shape ({batch_size}, {embedding_dim}), but got {z_q.shape}"

    # 2. Check the shape of the indices
    assert indices.shape == (batch_size,), \
        f"Expected indices shape ({batch_size},), but got {indices.shape}"

def test_vq_ste_forward_pass_value(vq_setup):
    """
    Verifies the value of the Straight-Through Estimator in the forward pass.
    The output z_q should be numerically identical to the pure quantized vector.
    """
    quantizer = vq_setup["quantizer"]
    z = vq_setup["z"]

    # z_q_ste is the output of the forward pass, which includes the STE trick
    z_q_ste, indices, _, _ = quantizer(z)
    
    # z_q_pure is the actual closest vector from the codebook, found by direct lookup
    z_q_pure = quantizer.embedding(indices)

    # The STE is formulated as z + (z_q_pure - z).detach().
    # In the forward pass (value computation), this is numerically equivalent to z_q_pure
    # because the values of z and z.detach() are the same.
    # Therefore, the output of the layer should be numerically identical to the pure codebook vector.
    assert torch.allclose(z_q_ste, z_q_pure, atol=1e-6), \
        "The output of the STE should be numerically identical to the pure quantized vector in the forward pass."

def test_vq_indices_validity(vq_setup):
    """
    Tests if the returned indices are within the valid range [0, num_embeddings-1].
    """
    quantizer = vq_setup["quantizer"]
    z = vq_setup["z"]
    num_embeddings = vq_setup["num_embeddings"]

    _, indices, _, _ = quantizer(z)

    # Check if all indices are non-negative
    assert torch.all(indices >= 0), "Found negative indices."
    # Check if all indices are less than the number of embeddings
    assert torch.all(indices < num_embeddings), "Found indices greater than or equal to num_embeddings."

def test_vq_losses_are_non_negative(vq_setup):
    """
    Tests if the commitment and codebook losses are non-negative scalars.
    """
    quantizer = vq_setup["quantizer"]
    z = vq_setup["z"]

    _, _, commitment_loss, codebook_loss = quantizer(z)

    assert commitment_loss.item() >= 0, f"Commitment loss must be non-negative, but got {commitment_loss.item()}"
    assert codebook_loss.item() >= 0, f"Codebook loss must be non-negative, but got {codebook_loss.item()}"
    assert commitment_loss.dim() == 0, "Commitment loss should be a scalar."
    assert codebook_loss.dim() == 0, "Codebook loss should be a scalar."

def test_vq_closest_vector_logic(vq_setup):
    """
    Explicitly tests if the returned index corresponds to the closest codebook vector.
    """
    quantizer = vq_setup["quantizer"]
    z = vq_setup["z"]
    codebook = quantizer.embedding.weight

    # Get the output index from the module
    _, indices, _, _ = quantizer(z)

    # For the first vector in the batch, manually find the closest codebook vector
    first_z = z[0]
    
    # Manually compute Euclidean distances
    manual_dists = torch.norm(codebook - first_z, p=2, dim=1)
    manual_closest_index = torch.argmin(manual_dists)

    # Compare the module's output with the manual calculation
    assert indices[0] == manual_closest_index, \
        f"Module index {indices[0]} does not match manually calculated closest index {manual_closest_index}."

def test_vq_straight_through_estimator_gradient(vq_setup):
    """
    Verifies that gradients flow back to the input 'z' despite the non-differentiable argmin.
    """
    quantizer = vq_setup["quantizer"]
    z = vq_setup["z"]
    z.requires_grad = True # Ensure gradient tracking is on for z

    z_q, _, _, _ = quantizer(z)

    # The output z_q should have a gradient function because of the STE
    assert z_q.grad_fn is not None, "z_q should have a grad_fn due to the straight-through estimator."

    # Perform a backward pass
    z_q.sum().backward()

    # Check if the input 'z' received a gradient
    assert z.grad is not None, "Input 'z' should have received a gradient."
    # The gradient for the STE should be 1, so the gradient of the sum should be a tensor of ones
    assert torch.allclose(z.grad, torch.ones_like(z)), "Gradient of z is not what was expected from STE."

# --- Test Fixture for ResidualQuantizer ---
@pytest.fixture
def rq_setup():
    """Provides a standard ResidualQuantizer instance and input tensor for tests."""
    num_layers = 4
    num_embeddings = 64
    embedding_dim = 16
    batch_size = 8
    
    rq = ResidualQuantizer(num_layers, num_embeddings, embedding_dim)
    z = torch.randn(batch_size, embedding_dim)
    
    return {
        "rq": rq,
        "z": z,
        "num_layers": num_layers,
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size
    }

# --- Tests for the ResidualQuantizer Class ---

def test_rq_forward_pass_shapes(rq_setup):
    """
    Tests if the RQ forward pass produces tensors of the correct shape.
    """
    rq = rq_setup["rq"]
    z = rq_setup["z"]
    batch_size = rq_setup["batch_size"]
    embedding_dim = rq_setup["embedding_dim"]
    num_layers = rq_setup["num_layers"]

    z_q_final, sid_tokens, _ = rq(z)

    # 1. Check the shape of the final quantized vector
    assert z_q_final.shape == (batch_size, embedding_dim), \
        f"Expected z_q_final shape ({batch_size}, {embedding_dim}), but got {z_q_final.shape}"

    # 2. Check the shape of the Semantic ID tokens
    assert sid_tokens.shape == (batch_size, num_layers), \
        f"Expected sid_tokens shape ({batch_size}, {num_layers}), but got {sid_tokens.shape}"

def test_rq_sid_token_validity(rq_setup):
    """
    Tests if all tokens in the generated SIDs are within the valid range.
    """
    rq = rq_setup["rq"]
    z = rq_setup["z"]
    num_embeddings = rq_setup["num_embeddings"]

    _, sid_tokens, _ = rq(z)

    assert torch.all(sid_tokens >= 0), "Found negative SID tokens."
    assert torch.all(sid_tokens < num_embeddings), "Found SID tokens greater than or equal to num_embeddings."

def test_rq_reconstruction_property(rq_setup):
    """
    Verifies that z_q_final is the sum of the intermediate quantized vectors.
    This test requires modifying the RQ forward pass to return intermediate vectors.
    """
    rq = rq_setup["rq"]
    z = rq_setup["z"]

    # We need to get the intermediate vectors, so we'll call the forward pass and then access them.
    # A more robust way would be to modify the forward pass to return them for testing.
    # For now, we'll re-implement the loop logic here for verification.
    
    quantized_vectors = []
    residual = z.clone()
    for quantizer in rq.quantizers:
        z_q_k, _, _, _ = quantizer(residual)
        # The residual for the next step is calculated with the detached z_q_k
        residual = residual - z_q_k.detach()
        # We store the z_q_k that still has gradient info for the final sum
        quantized_vectors.append(z_q_k)

    z_q_final_manual = sum(quantized_vectors)
    z_q_final_from_module, _, _ = rq(z)

    assert torch.allclose(z_q_final_from_module, z_q_final_manual, atol=1e-6), \
        "Final quantized vector from module does not match the manual sum of intermediate vectors."

def test_rq_loss_aggregation(rq_setup):
    """
    Tests if the total loss is correctly aggregated from all VQ layers.
    """
    rq = rq_setup["rq"]
    z = rq_setup["z"]
    
    _, _, total_loss = rq(z)
    
    # Manually calculate the loss
    manual_commitment_loss = 0
    manual_codebook_loss = 0
    residual = z.clone()
    for quantizer in rq.quantizers:
        z_q_k, _, c_loss, cb_loss = quantizer(residual)
        residual = residual - z_q_k.detach()
        manual_commitment_loss += c_loss
        manual_codebook_loss += cb_loss
    
    manual_total_loss = manual_commitment_loss + manual_codebook_loss

    assert torch.isclose(total_loss, manual_total_loss), \
        f"Aggregated loss {total_loss.item()} does not match manual sum {manual_total_loss.item()}."
