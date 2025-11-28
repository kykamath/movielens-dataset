import os
import queue
import time
from dotenv import load_dotenv
from huggingface_hub import login

from models import Movie, HUB_ENRICHED_REPO_ID, ENRICHED_MOVIES_JSONL
from enrichment_utils import (
    get_movie_details_batch,
    load_processed_movie_ids,
    build_movie_queue,
    append_movie_to_jsonl,
    upload_enriched_dataset,
    sync_enriched_data_from_hub
)


def main():
    # --- Configuration ---
    MOVIES_TO_PROCESS = 2500  # Total number of movies we want to process in this crawl.
    BATCH_SIZE = 10           # Number of movies we want to enrich in a single OpenAI API call.
    OUTPUT_FILE = ENRICHED_MOVIES_JSONL
    MOVIES_CSV_PATH = "ml-32m/movies.csv" # Path to the original MovieLens CSV

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

    # --- 2. Sync with Hugging Face Hub ---
    # If a local file exists, it will be removed and replaced with the latest from the Hub.
    # If the Hub repo doesn't exist, it will proceed with an empty local file.
    if can_upload:
        sync_enriched_data_from_hub(HUB_ENRICHED_REPO_ID, OUTPUT_FILE)
    else:
        print("Skipping sync from Hub as login could not be completed.")

    # --- 3. Load Processed IDs and Build Queue ---
    processed_movie_ids = load_processed_movie_ids(OUTPUT_FILE)
    print(f"Found {len(processed_movie_ids)} processed movies from '{OUTPUT_FILE}'.")

    movie_queue = build_movie_queue(MOVIES_CSV_PATH, processed_movie_ids)
    print(f"Queue built with {movie_queue.qsize()} new movies to process.")

    # --- 4. Process Movies in Batches ---
    processed_count = 0
    while not movie_queue.empty() and processed_count < MOVIES_TO_PROCESS:
        batch: List[Movie] = []
        current_batch_size = min(BATCH_SIZE, MOVIES_TO_PROCESS - processed_count, movie_queue.qsize())
        
        if current_batch_size <= 0:
            break
            
        for _ in range(current_batch_size):
            if not movie_queue.empty():
                batch.append(movie_queue.get())

        if not batch:
            continue

        try:
            movie_titles_in_batch = [movie.title for movie in batch]
            print(f"Processing batch of {len(batch)} movies: {movie_titles_in_batch}")
            
            get_movie_details_batch(batch) # Enrich movies in batch

            for movie in batch:
                append_movie_to_jsonl(movie, OUTPUT_FILE)
                processed_count += 1

            print(f"Batch processed. Total movies processed in this run: {processed_count}/{MOVIES_TO_PROCESS}")

        except Exception as e:
            print(f"Failed to process batch. Error: {e}")
        
        print(f"{movie_queue.qsize()} movies remaining in the queue.")
        time.sleep(1) # Be kind to the API

    print("\nMovie enrichment complete.")

    # --- 5. Upload Final Enriched Dataset to Hugging Face Hub ---
    if can_upload:
        upload_enriched_dataset(OUTPUT_FILE, repo_id=HUB_ENRICHED_REPO_ID, private=False)
    else:
        print("Skipping upload to Hugging Face Hub as login could not be completed.")

if __name__ == "__main__":
    main()
