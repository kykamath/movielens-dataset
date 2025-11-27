import pandas as pd
from datasets import Dataset
from huggingface_hub import login
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import queue
import time
from dataclasses import dataclass, field, asdict
from typing import List, Set, Dict


@dataclass
class Movie:
    movie_id: int
    title: str
    genres: List[str]
    plot_summary: str = ""
    director: str = ""
    stars: List[str] = field(default_factory=list)


def get_movie_details_batch(movies: List[Movie]) -> None:
    """
    Generates plot summary, director, and stars for a batch of movies using the OpenAI API.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    movie_titles_str = "\n".join([f"{i+1}. {movie.title}" for i, movie in enumerate(movies)])

    prompt = f"""
    [INST] You are an expert film critic. Given a numbered list of movie titles, provide a JSON object containing a list of movie details.
    Each item in the list should be a JSON object with the following fields for the corresponding movie:
    - "title": The original movie title.
    - "plot_summary": A brief, but comprehensive plot summary (100-150 words).
    - "director": The name of the director.
    - "stars": A list of the top 3 starring actors.

    The output should be a single JSON object with a key "movies" that contains the list of movie details.
    Ensure the order of movies in the output list matches the order of the input titles.

    Movie Titles:
    {movie_titles_str}
    [/INST]
    """

    max_tokens = len(movies) * 400

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in movies and respond in JSON format."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.95,
        response_format={"type": "json_object"},
    )

    response_data = json.loads(response.choices[0].message.content)
    details_list = response_data.get("movies", [])

    movie_map: Dict[str, Movie] = {movie.title: movie for movie in movies}

    for details in details_list:
        title = details.get("title")
        if title in movie_map:
            movie = movie_map[title]
            movie.plot_summary = details.get("plot_summary", "")
            movie.director = details.get("director", "")
            movie.stars = details.get("stars", [])
        else:
            print(f"Warning: Received details for title '{title}' which was not in the request batch.")


def load_movies_dataset(csv_path: str = "ml-32m/movies.csv") -> Dataset:
    """
    Read movies.csv and create a Hugging Face dataset.
    """
    df = pd.read_csv(csv_path)
    df['genres'] = df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    return Dataset.from_pandas(df)


def load_processed_movie_ids(output_path: str) -> Set[int]:
    """
    Loads processed movie IDs from the output JSONL file.
    """
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    movie_data = json.loads(line)
                    if 'movie_id' in movie_data:
                        processed_ids.add(movie_data['movie_id'])
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from line: {line.strip()}")
    return processed_ids


def build_movie_queue(csv_path: str, processed_ids: Set[int]) -> queue.Queue:
    """
    Loads movies from a CSV file and builds a queue of unprocessed Movie objects.
    """
    dataset = load_movies_dataset(csv_path)
    movie_queue = queue.Queue()
    for movie_data in dataset:
        if movie_data['movieId'] not in processed_ids:
            movie = Movie(
                movie_id=movie_data['movieId'],
                title=movie_data['title'],
                genres=movie_data['genres']
            )
            movie_queue.put(movie)
    return movie_queue


def append_movie_to_jsonl(movie: Movie, output_path: str) -> None:
    """
    Appends a movie object to a JSONL file.
    """
    movie_dict = asdict(movie)
    with open(output_path, 'a') as f:
        f.write(json.dumps(movie_dict) + '\n')


def push_dataset_to_hub(dataset: Dataset, repo_id: str, private: bool = False) -> None:
    """
    Push the dataset to Hugging Face Hub.
    """
    dataset.push_to_hub(repo_id, private=private)
    print(f"Dataset pushed to Hugging Face Hub: https://huggingface.co/datasets/{repo_id}")


def upload_enriched_dataset(jsonl_path: str, repo_id: str, private: bool = False) -> None:
    """
    Loads a JSONL file and pushes it to the Hugging Face Hub.
    """
    if not os.path.exists(jsonl_path):
        print(f"Error: JSONL file not found at {jsonl_path}. Cannot upload.")
        return

    print(f"Loading dataset from {jsonl_path}...")
    enriched_dataset = Dataset.from_json(jsonl_path)

    print(f"Pushing dataset to {repo_id}...")
    push_dataset_to_hub(enriched_dataset, repo_id, private)


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

    # Name of the dataset on Hugging Face Hub.
    HUB_ENRICHED_REPO_ID = "krishnakamath/movielens-32m-movies-enriched"
    # Total number of movies we want to process in this crawl.
    MOVIES_TO_PROCESS = 250
    # Number of movies we want to enrich in a single OpenAI API call.
    BATCH_SIZE = 10

    output_file = "movies_with_details.jsonl"
    processed_movie_ids = load_processed_movie_ids(output_file)
    print(f"Found {len(processed_movie_ids)} processed movies.")

    movie_queue = build_movie_queue("ml-32m/movies.csv", processed_movie_ids)
    print(f"Queue built with {movie_queue.qsize()} movies to process.")

    processed_count = 0
    while not movie_queue.empty() and processed_count < MOVIES_TO_PROCESS:
        batch = []
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
            
            get_movie_details_batch(batch)

            for movie in batch:
                append_movie_to_jsonl(movie, output_file)
                processed_count += 1

            print(f"Batch processed. Total movies processed: {processed_count}/{MOVIES_TO_PROCESS}")

        except Exception as e:
            print(f"Failed to process batch. Error: {e}")
        
        print(f"{movie_queue.qsize()} movies remaining in the queue.")
        time.sleep(1)

    print("\nMovie processing complete.")
    if can_upload:
        upload_enriched_dataset(output_file, repo_id=HUB_ENRICHED_REPO_ID, private=False)
    else:
        print("Skipping upload to Hugging Face Hub as login could not be completed.")