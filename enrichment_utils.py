import pandas as pd
from datasets import Dataset, load_dataset, Features, Value, Sequence
import os
import json
from openai import OpenAI
import queue
import time
from dataclasses import asdict
from typing import List, Set, Dict

from models import Movie # Assuming Movie class is in models.py


def get_movie_details_batch(movies: List[Movie]) -> None:
    """
    Generates plot summary, director, and stars for a batch of movies using the OpenAI API.
    Updates the Movie objects in place.
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

    max_tokens = len(movies) * 400 # Adjust max_tokens based on batch size

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
            director = details.get("director", "")
            # Ensure director is always a string
            if isinstance(director, list):
                movie.director = ", ".join(director)
            else:
                movie.director = director
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
    Loads movies from a CSV file and builds a queue of Movie objects that have not been processed yet.
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


def upload_enriched_dataset(jsonl_path: str, repo_id: str, private: bool = False) -> None:
    """
    Loads a JSONL file and pushes it to the Hugging Face Hub.
    This function now handles potential inconsistencies in the 'director' field.
    """
    if not os.path.exists(jsonl_path):
        print(f"Error: JSONL file not found at {jsonl_path}. Cannot upload.")
        return

    print(f"Loading and cleaning dataset from {jsonl_path}...")

    # Define a flexible schema to read the data without initial errors
    features = Features({
        'movie_id': Value('int64'),
        'title': Value('string'),
        'genres': Sequence(Value('string')),
        'plot_summary': Value('string'),
        'director': Value('string'), # Initially read as string, will handle lists manually
        'stars': Sequence(Value('string'))
    })

    # Manually read and clean the data
    cleaned_data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                # If director is a list, convert it to a comma-separated string
                if isinstance(record.get('director'), list):
                    record['director'] = ", ".join(record['director'])
                cleaned_data.append(record)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line: {line.strip()}")

    # Create a DataFrame and then a Dataset
    df = pd.DataFrame(cleaned_data)
    enriched_dataset = Dataset.from_pandas(df, features=features)

    print(f"Pushing dataset to {repo_id}...")
    enriched_dataset.push_to_hub(repo_id, private=private)
    print(f"Dataset pushed to Hugging Face Hub: https://huggingface.co/datasets/{repo_id}")


def sync_enriched_data_from_hub(repo_id: str, output_path: str) -> None:
    """
    Downloads the latest version of the dataset from Hugging Face Hub, deleting the old local file first.
    """
    if os.path.exists(output_path):
        print(f"Removing existing local file: {output_path}")
        os.remove(output_path)

    try:
        print(f"Attempting to download latest dataset from '{repo_id}'...")
        hub_dataset = load_dataset(repo_id, split="train")
        print(f"Successfully downloaded dataset. Found {len(hub_dataset)} records.")

        print(f"Syncing dataset to new local file: {output_path}")
        with open(output_path, 'w') as f:
            for movie_data in hub_dataset:
                # Ensure only relevant fields are written if schema changed
                movie_to_write = {
                    "movie_id": movie_data.get('movie_id'),
                    "title": movie_data.get('title'),
                    "genres": movie_data.get('genres', []),
                    "plot_summary": movie_data.get('plot_summary', ''),
                    "director": movie_data.get('director', ''),
                    "stars": movie_data.get('stars', [])
                }
                f.write(json.dumps(movie_to_write) + '\n')
        print("Local file synced with Hugging Face Hub.")

    except Exception as e:
        print(f"An error occurred while trying to download the dataset: {e}")
        print("Proceeding without a synced file. A new local file will be created if it doesn't exist.")
