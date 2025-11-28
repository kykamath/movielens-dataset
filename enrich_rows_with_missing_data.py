import os
import time
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
import json
from dataclasses import asdict

from models import Movie, HUB_ENRICHED_REPO_ID
from enrichment_utils import get_movie_details_batch, upload_enriched_dataset

def main():
    # --- Configuration ---
    BATCH_SIZE = 10
    OUTPUT_FILE = "movies_with_details_updated.jsonl"

    # --- 1. Authentication ---
    load_dotenv()
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    can_upload = False
    if hf_token:
        print("Logging in to Hugging Face Hub...")
        login(token=hf_token)
        can_upload = True
    else:
        print("Warning: HUGGING_FACE_HUB_TOKEN not found. Upload to Hub will be skipped.")

    # --- 2. Load Data from Hugging Face Hub ---
    print(f"Loading dataset from '{HUB_ENRICHED_REPO_ID}'...")
    try:
        hub_dataset = load_dataset(HUB_ENRICHED_REPO_ID, split="train")
        print(f"Successfully downloaded dataset. Found {len(hub_dataset)} records.")
    except Exception as e:
        print(f"Failed to load dataset from Hub: {e}")
        return

    # --- 3. Identify Movies with Missing Details and Preserve All Movies ---
    movies_to_enrich = []
    all_movies = []
    for record in hub_dataset:
        movie = Movie(**record)
        all_movies.append(movie)
        if not movie.plot_summary or not movie.director or not movie.stars:
            movies_to_enrich.append(movie)

    print(f"Found {len(movies_to_enrich)} movies with missing details (plot, director, or stars).")

    if not movies_to_enrich:
        print("No movies to enrich. Exiting.")
        return

    # --- 4. Process Movies in Batches ---
    enriched_count = 0
    for i in range(0, len(movies_to_enrich), BATCH_SIZE):
        batch = movies_to_enrich[i:i + BATCH_SIZE]

        try:
            movie_titles_in_batch = [movie.title for movie in batch]
            print(f"Processing batch of {len(batch)} movies: {movie_titles_in_batch}")

            get_movie_details_batch(batch)  # Enrich movies in batch

            enriched_count += len(batch)
            print(f"Batch processed. Total movies enriched so far: {enriched_count}/{len(movies_to_enrich)}")

        except Exception as e:
            print(f"Failed to process batch. Error: {e}")

        time.sleep(1)  # Be kind to the API

    print("\nMovie enrichment complete.")

    # --- 5. Save Updated Dataset to new JSONL file ---
    print(f"Saving updated dataset to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, 'w') as f:
        for movie in all_movies:
            f.write(json.dumps(asdict(movie)) + '\n')
    print("Saved updated dataset.")


    # --- 6. Upload Final Enriched Dataset to Hugging Face Hub ---
    if can_upload:
        upload_enriched_dataset(OUTPUT_FILE, repo_id=HUB_ENRICHED_REPO_ID, private=False)
    else:
        print("Skipping upload to Hugging Face Hub as login could not be completed.")


if __name__ == "__main__":
    main()
