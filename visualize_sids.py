import torch
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from datasets import load_dataset
from models import HUB_EMBEDDINGS_REPO_ID
from create_semantic_ids import RQVAE # We need to load our LightningModule

def visualize_semantic_ids(checkpoint_path: str, embeddings_tensor: torch.Tensor, movie_metadata: pd.DataFrame):
    """
    Generates and visualizes Semantic IDs using UMAP.

    Args:
        checkpoint_path: Path to the best trained RQ-VAE model checkpoint.
        embeddings_tensor: The full tensor of movie embeddings.
        movie_metadata: A DataFrame with movie titles and other info, indexed to match the tensor.
    """
    print("Loading best model from checkpoint...")
    model = RQVAE.load_from_checkpoint(checkpoint_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Generating Semantic IDs for all movies...")
    with torch.no_grad():
        _, ids, _ = model(embeddings_tensor.to(device))
        all_semantic_ids = ids.cpu().numpy()

    print("Running UMAP for dimensionality reduction... (This may take a minute)")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings_tensor.cpu().numpy())

    # --- Create a DataFrame for plotting ---
    # This combines UMAP coordinates, movie metadata, and the learned SIDs
    df = movie_metadata.copy()
    df['umap_x'] = embeddings_2d[:, 0]
    df['umap_y'] = embeddings_2d[:, 1]
    
    # Extract each token into its own column for easier filtering and coloring
    for i in range(all_semantic_ids.shape[1]):
        df[f'T{i+1}'] = all_semantic_ids[:, i]
    
    # Create a string representation of the full SID for hover data
    df['SID'] = df.apply(lambda row: ' '.join([f"<T{i+1}:{int(row[f'T{i+1}']):04d}>" for i in range(model.hparams.num_layers)]), axis=1)

    # --- Create the Interactive Visualization ---
    print("Generating interactive plot with Plotly...")
    fig = px.scatter(
        df,
        x='umap_x',
        y='umap_y',
        color=df['T1'].astype(str),  # Color by the first token (T1)
        hover_name='title',         # Show movie title on hover
        hover_data=['genres', 'SID'], # Show genres and full SID in the hover tooltip
        title="UMAP Visualization of Movie Embeddings, Colored by Semantic ID Token 1 (T1)"
    )

    fig.update_layout(
        legend_title_text='First Semantic Token (T1)',
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2"
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))

    # Save to an HTML file
    output_filename = "semantic_id_visualization.html"
    fig.write_html(output_filename)
    print(f"\n✅ Visualization saved to '{output_filename}'. Open this file in your browser.")

if __name__ == '__main__':
    # --- 1. Load Data ---
    print(f"Loading data from Hugging Face Hub: {HUB_EMBEDDINGS_REPO_ID}")
    hub_dataset = load_dataset(HUB_EMBEDDINGS_REPO_ID, split="train")
    
    # Create a metadata DataFrame and a corresponding tensor
    metadata = []
    embeddings_list = []
    for item in hub_dataset:
        if item['all_mpnet_base_v2_embedding']:
            embeddings_list.append(item['all_mpnet_base_v2_embedding'])
            metadata.append({
                'title': item['title'],
                'genres': ', '.join(item['genres'])
            })

    if not embeddings_list:
        print("No embeddings found. Exiting.")
    else:
        full_embeddings_tensor = torch.tensor(embeddings_list, dtype=torch.float32)
        metadata_df = pd.DataFrame(metadata)
        
        # --- 2. Specify Checkpoint and Run Visualization ---
        # IMPORTANT: Replace this with the actual path to your best model checkpoint
        best_checkpoint_path = "checkpoints/rqvae-best-model-epoch=99-val_loss=0.0039.ckpt" # Example path
        
        try:
            visualize_semantic_ids(best_checkpoint_path, full_embeddings_tensor, metadata_df)
        except FileNotFoundError:
            print(f"\n❌ ERROR: Checkpoint file not found at '{best_checkpoint_path}'.")
            print("Please update the 'best_checkpoint_path' variable with the correct path from your 'checkpoints' directory.")

