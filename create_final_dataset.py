import torch
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from dotenv import load_dotenv
import os
from models import Movie, HUB_ENRICHED_REPO_ID, HUB_MODEL_ID, HUB_SIDS_DATASET_ID, EMBEDDING_MODEL_NAME
from residual_quantized_vae import ResidualQuantizer

def main():
    """
    This script creates the final, comprehensive dataset by:
    1. Loading the base enriched text data.
    2. Loading the pre-trained RQ-VAE model.
    3. Generating embeddings for the text data on the fly.
    4. Using the RQ-VAE model to generate Semantic IDs from the embeddings.
    5. Adding both the embeddings and the SIDs to the dataset and uploading it to the Hub.
    """
    # --- 1. Authentication ---
    load_dotenv()
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("Logging in to Hugging Face Hub...")
        login(token=hf_token)
    else:
        print("Warning: HUGGING_FACE_HUB_TOKEN not found. Cannot upload dataset.")
        return

    # --- 2. Load Models and Source Data ---
    print(f"Loading pre-trained RQ-VAE model from Hugging Face Hub: {HUB_MODEL_ID}")
    try:
        rq_model = ResidualQuantizer.from_pretrained(HUB_MODEL_ID)
    except Exception as e:
        print(f"Could not load model from Hub. Error: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rq_model = rq_model.to(device)
    rq_model.eval()
    print(f"Using device: {device}")

    print(f"Loading source enriched dataset from: {HUB_ENRICHED_REPO_ID}")
    source_dataset = load_dataset(HUB_ENRICHED_REPO_ID, split="train")

    # --- 3. Generate Embeddings ---
    print(f"Generating embeddings using '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    texts_to_embed = []
    items_with_plots_indices = []
    
    for i, item in enumerate(source_dataset):
        if item.get('plot_summary'):
            # Manually map fields to handle schema mismatches and ignore old columns
            movie = Movie(
                movie_id=item.get('movie_id'),
                title=item.get('title'),
                genres=item.get('genres', []),
                plot_summary=item.get('plot_summary', ''),
                director=item.get('director', ''),
                stars=item.get('stars', [])
                # We do not load any embedding field here, as we are about to generate a new one
            )
            texts_to_embed.append(movie.to_embedding_string())
            items_with_plots_indices.append(i)

    if not texts_to_embed:
        print("No valid texts found to generate embeddings.")
        return

    embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=True, device=device)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)

    # --- 4. Generate Semantic IDs ---
    print("Generating Semantic IDs from embeddings...")
    with torch.no_grad():
        _, sids_tensor, _ = rq_model(embeddings_tensor)
        all_semantic_ids = sids_tensor.cpu().numpy().tolist()

    # --- 5. Combine and Upload Final Dataset ---
    print("Combining original data with new embeddings and Semantic IDs...")
    
    movie_id_to_embedding = {source_dataset[i]['movie_id']: emb.tolist() for i, emb in zip(items_with_plots_indices, embeddings)}
    movie_id_to_sid = {source_dataset[i]['movie_id']: sid for i, sid in zip(items_with_plots_indices, all_semantic_ids)}

    def add_embeddings_and_sids(example):
        movie_id = example['movie_id']
        example['all_mpnet_base_v2_embedding'] = movie_id_to_embedding.get(movie_id, [])
        example['semantic_id'] = movie_id_to_sid.get(movie_id, [])
        return example

    # Use .map() for a clean and efficient update
    final_dataset = source_dataset.map(add_embeddings_and_sids)
    
    # Remove the old, now-irrelevant embedding column if it exists
    if 'all_MiniLM_L12_v2_embedding' in final_dataset.column_names:
        final_dataset = final_dataset.remove_columns(['all_MiniLM_L12_v2_embedding'])

    print(f"\nUploading final dataset with SIDs to: {HUB_SIDS_DATASET_ID}")
    final_dataset.push_to_hub(HUB_SIDS_DATASET_ID, private=False)
    
    print("\n✅ Process Complete. Final dataset is available on the Hugging Face Hub.")
    print(f"Dataset URL: https://huggingface.co/datasets/{HUB_SIDS_DATASET_ID}")

if __name__ == '__main__':
    main()
