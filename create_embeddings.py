from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from models import Movie, HUB_ENRICHED_REPO_ID, HUB_EMBEDDINGS_REPO_ID
from datasets import load_dataset, Dataset
from scipy.spatial.distance import cosine # Import for cosine similarity
import os
import json
from huggingface_hub import login
from dotenv import load_dotenv


def generate_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Generates embeddings for a list of texts using the 'all-MiniLM-L12-v2' model.
    The sentence-transformers library handles batching internally, so you can pass a large list of texts directly.

    Args:
        texts: A list of strings to be encoded.
        batch_size: The batch size to use for the computation. A larger batch size can be faster but uses more VRAM.

    Returns:
        A numpy array containing the embeddings.
    """
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings

def load_movies_from_hub() -> List[Movie]:
    """
    Loads the enriched movie dataset from Hugging Face Hub and converts each entry to a Movie object.

    Returns:
        A list of Movie objects.
    """
    print(f"Loading dataset from '{HUB_ENRICHED_REPO_ID}'...")
    dataset = load_dataset(HUB_ENRICHED_REPO_ID, split="train")
    
    movies = []
    for item in dataset:
        movie = Movie(
            movie_id=item.get('movie_id'),
            title=item.get('title'),
            genres=item.get('genres', []),
            plot_summary=item.get('plot_summary', ''),
            director=item.get('director', ''),
            stars=item.get('stars', []),
            all_MiniLM_L12_v2_embedding=item.get('all_MiniLM_L12_v2_embedding', []) # Ensure this field is handled
        )
        movies.append(movie)
        
    print(f"Successfully loaded and converted {len(movies)} movies.")
    return movies

def save_movies_to_jsonl(movies: List[Movie], output_path: str) -> None:
    """
    Saves a list of Movie objects to a JSONL file.
    """
    print(f"Saving {len(movies)} movies to {output_path}...")
    with open(output_path, 'w') as f:
        for movie in movies:
            # Convert Movie dataclass to dictionary for JSON serialization
            movie_dict = {
                "movie_id": movie.movie_id,
                "title": movie.title,
                "genres": movie.genres,
                "plot_summary": movie.plot_summary,
                "director": movie.director,
                "stars": movie.stars,
                "all_MiniLM_L12_v2_embedding": movie.all_MiniLM_L12_v2_embedding
            }
            f.write(json.dumps(movie_dict) + '\n')
    print(f"Movies saved to {output_path}.")

def upload_embeddings_dataset(jsonl_path: str, repo_id: str) -> None:
    """
    Loads a JSONL file and pushes it to the Hugging Face Hub.
    """
    if not os.path.exists(jsonl_path):
        print(f"Error: JSONL file not found at {jsonl_path}. Cannot upload.")
        return

    print(f"Loading dataset from {jsonl_path} for upload...")
    dataset_to_upload = Dataset.from_json(jsonl_path)
    
    print(f"Pushing dataset to {repo_id}...")
    dataset_to_upload.push_to_hub(repo_id, private=False)
    print(f"Dataset pushed to Hugging Face Hub: https://huggingface.co/datasets/{repo_id}")


def verify_embeddings(movies: List[Movie]):
    """
    Performs technical and semantic validation of generated embeddings.
    """
    print("\n--- Verifying Embeddings ---")

    if not movies:
        print("No movies to verify.")
        return

    # Technical Sanity Checks
    print("\n1. Technical Sanity Checks:")
    all_embeddings_valid = True
    for i, movie in enumerate(movies):
        embedding = np.array(movie.all_MiniLM_L12_v2_embedding)
        
        if embedding.shape != (384,):
            print(f"  ERROR: Movie '{movie.title}' (ID: {movie.movie_id}) has incorrect embedding dimension: {embedding.shape}")
            all_embeddings_valid = False
        if np.all(embedding == 0):
            print(f"  ERROR: Movie '{movie.title}' (ID: {movie.movie_id}) has a zero vector embedding.")
            all_embeddings_valid = False
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            print(f"  ERROR: Movie '{movie.title}' (ID: {movie.movie_id}) has NaN or Inf values in embedding.")
            all_embeddings_valid = False
        if i >= 100: # Limit checks to first 100 for brevity
            break
    
    if all_embeddings_valid:
        print("  All technical checks passed for sampled movies.")
    else:
        print("  Some technical checks failed. Review the errors above.")

    # Semantic Validation (Cosine Similarity)
    print("\n2. Semantic Validation (Cosine Similarity):")
    
    # Define test cases (replace with actual movie titles from your dataset if available)
    test_cases = [
        ("Toy Story (1995)", "Toy Story 2 (1999)", "similar"),
        ("Toy Story (1995)", "Schindler's List (1993)", "dissimilar"),
        ("Dr. No (1962)", "Goldfinger (1964)", "similar"),
        ("Schindler's List (1993)", "Goldfinger (1964)", "dissimilar"),
        ("Thunderball (1965)", "From Russia with Love (1963)", "similar"),
    ]

    movie_lookup = {movie.title: movie for movie in movies if movie.all_MiniLM_L12_v2_embedding}

    for title1, title2, expected_relation in test_cases:
        movie1 = movie_lookup.get(title1)
        movie2 = movie_lookup.get(title2)

        if movie1 and movie2:
            emb1 = np.array(movie1.all_MiniLM_L12_v2_embedding)
            emb2 = np.array(movie2.all_MiniLM_L12_v2_embedding)
            
            # Ensure embeddings are not zero vectors before calculating similarity
            if np.all(emb1 == 0) or np.all(emb2 == 0):
                print(f"  Skipping similarity for '{title1}' and '{title2}' due to zero embedding.")
                continue

            similarity = 1 - cosine(emb1, emb2) # Cosine distance is 1 - cosine similarity
            print(f"  Similarity between '{title1}' and '{title2}' ({expected_relation}): {similarity:.4f}")
            
            # Basic assertion for expected ranges
            if expected_relation == "similar" and similarity < 0.5: # Threshold can be adjusted
                print(f"    WARNING: Expected high similarity but got {similarity:.4f}")
            elif expected_relation == "dissimilar" and similarity > 0.4: # Threshold can be adjusted
                print(f"    WARNING: Expected low similarity but got {similarity:.4f}")
        else:
            print(f"  Could not find one or both movies for similarity check: '{title1}', '{title2}'")

    print("\n--- Embedding Verification Complete ---")


if __name__ == "__main__":
    load_dotenv()

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    can_upload = False
    if hf_token:
        print("Logging in to Hugging Face Hub...")
        login(token=hf_token)
        can_upload = True
    else:
        print("Warning: HUGGING_FACE_HUB_TOKEN not found. Upload to Hub will be skipped.")

    output_embeddings_file = "movies_with_embeddings.jsonl"

    movies = load_movies_from_hub()

    if movies:
        movies_to_embed = [movie for movie in movies if movie.plot_summary]
        
        if movies_to_embed:
            texts_to_embed = [movie.to_embedding_string() for movie in movies_to_embed]
            
            print(f"\nGenerating embeddings for {len(texts_to_embed)} movies...")
            embeddings = generate_embeddings(texts_to_embed)

            for movie, embedding in zip(movies_to_embed, embeddings):
                movie.all_MiniLM_L12_v2_embedding = embedding.tolist()

            print(f"\nSuccessfully generated and assigned embeddings.")
            print("Shape of the full embeddings matrix:", embeddings.shape)
            
            verify_embeddings(movies_to_embed)

            # save_movies_to_jsonl(movies_to_embed, output_embeddings_file)

            # if can_upload:
            #     upload_embeddings_dataset(output_embeddings_file, repo_id=HUB_EMBEDDINGS_REPO_ID)
            # else:
            #     print("Skipping upload to Hugging Face Hub as login could not be completed.")

        else:
            print("No plot summaries found in the dataset to generate embeddings.")
    else:
        print("No movies loaded from the Hub.")