import torch
from rqvae_lightning import RQVAE
from models import HUB_EMBEDDINGS_REPO_ID
from datasets import load_dataset

# --- Configuration ---
# The ID of the trained model on the Hugging Face Hub
HUB_MODEL_ID = "krishnakamath/rq-vae-movielens" 

def main():
    """
    Loads a trained RQ-VAE model from the Hub and uses it to generate
    Semantic IDs for the movie embeddings dataset.
    """
    print(f"Loading pre-trained RQ-VAE model from Hugging Face Hub: {HUB_MODEL_ID}")
    
    # Load the LightningModule. This will automatically download the weights and config.
    # Note: We are loading the full LightningModule wrapper.
    try:
        model = RQVAE.load_from_hub(HUB_MODEL_ID)
    except Exception as e:
        print(f"Could not load model from Hub. Error: {e}")
        print("Please ensure the model exists and you have the correct permissions.")
        return

    # Set the model to evaluation mode
    model.eval()
    
    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # --- Load the dataset to be encoded ---
    print(f"Loading embeddings from Hugging Face Hub: {HUB_EMBEDDINGS_REPO_ID}")
    hub_dataset = load_dataset(HUB_EMBEDDINGS_REPO_ID, split="train")
    
    embeddings_list = [item['all_mpnet_base_v2_embedding'] for item in hub_dataset if item['all_mpnet_base_v2_embedding']]
    if not embeddings_list:
        print("No embeddings found in the dataset. Exiting.")
        return
        
    full_embeddings_tensor = torch.tensor(embeddings_list, dtype=torch.float32)
    print(f"Loaded {len(full_embeddings_tensor)} embeddings.")

    # --- Generate Semantic IDs ---
    print("\nGenerating Semantic IDs for all movies...")
    all_semantic_ids = []
    with torch.no_grad():
        # The forward pass of our LightningModule calls the quantizer
        _, ids, _ = model(full_embeddings_tensor.to(device))
        all_semantic_ids = ids.cpu().numpy()

    # --- Display Example Output ---
    print("\n--- Example Semantic IDs from Trained Model ---")
    print(f"Total number of SIDs generated: {len(all_semantic_ids)}")
    
    num_layers = model.hparams.num_layers
    print(f"Each SID is a sequence of {num_layers} indices (tokens).\n")

    for i in range(10):
        sid = all_semantic_ids[i]
        # Dynamically create the SID string based on the number of layers
        sid_str = " ".join([f"<T{j+1}:{token:04d}>" for j, token in enumerate(sid)])
        print(f"Movie {i + 1} SID: {sid_str}")

if __name__ == '__main__':
    main()
