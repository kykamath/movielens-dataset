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
from typing import List, Set


@dataclass
class Movie:
    movie_id: int
    title: str
    genres: List[str]
    plot_summary: str = ""
    director: str = ""
    stars: List[str] = field(default_factory=list)


def get_movie_details(movie: Movie) -> None:
    """
    Generates a plot summary, director, and top 3 stars for a given movie and updates the movie object.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = f"""
    [INST] You are an expert film critic. Given a movie title, provide a JSON object with the following fields:
    - "plot_summary": A brief, but comprehensive plot summary (100-150 words).
    - "director": The name of the director.
    - "stars": A list of the top 3 starring actors.

    Do not include any other information.

    Movie Title: {movie.title}
    [/INST]
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in movies."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.7,
        top_p=0.95,
        response_format={"type": "json_object"},
    )

    details_json = json.loads(response.choices[0].message.content)
    
    movie.plot_summary = details_json.get("plot_summary", "")
    movie.director = details_json.get("director", "")
    movie.stars = details_json.get("stars", [])


def load_movies_dataset(csv_path: str = "ml-32m/movies.csv") -> Dataset:
    """
    Read movies.csv and create a Hugging Face dataset.
    
    Args:
        csv_path: Path to the movies.csv file
        
    Returns:
        A Hugging Face Dataset object with movie data
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert genres from pipe-separated string to list
    df['genres'] = df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    
    # Create Hugging Face dataset
    dataset = Dataset.from_pandas(df)
    
    return dataset


def load_processed_movie_ids(output_path: str) -> Set[int]:
    """
    Loads processed movie IDs from the output JSONL file.
    
    Args:
        output_path: The path to the output JSONL file.
        
    Returns:
        A set of movie_ids that have already been processed.
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
                    # Handle cases where a line is not valid JSON
                    print(f"Warning: Could not decode JSON from line: {line.strip()}")
    return processed_ids


def build_movie_queue(csv_path: str, processed_ids: Set[int]) -> queue.Queue:
    """
    Loads movies from a CSV file and builds a queue of Movie objects that have not been processed yet.
    
    Args:
        csv_path: Path to the movies.csv file.
        processed_ids: A set of movie_ids that have already been processed.
        
    Returns:
        A queue.Queue object populated with unprocessed Movie objects.
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
    
    Args:
        movie: The Movie object to append.
        output_path: The path to the output JSONL file.
    """
    movie_dict = asdict(movie)
    with open(output_path, 'a') as f:
        f.write(json.dumps(movie_dict) + '\n')


def push_dataset_to_hub(dataset: Dataset, repo_id: str, private: bool = False) -> None:
    """
    Push the dataset to Hugging Face Hub.
    
    Args:
        dataset: The Hugging Face Dataset to push
        repo_id: Repository ID in format "username/dataset_name"
        private: Whether the dataset should be private (default: False)
    """
    dataset.push_to_hub(repo_id, private=private)
    print(f"Dataset pushed to Hugging Face Hub: https://huggingface.co/datasets/{repo_id}")
    
if __name__ == "__main__":
    load_dotenv()
    output_file = "movies_with_details.jsonl"
    MOVIES_TO_PROCESS = 10
    
    processed_movie_ids = load_processed_movie_ids(output_file)
    print(f"Found {len(processed_movie_ids)} processed movies.")
    
    movie_queue = build_movie_queue("ml-32m/movies.csv", processed_movie_ids)
    print(f"Queue built with {movie_queue.qsize()} movies to process.")

    processed_count = 0
    while not movie_queue.empty() and processed_count < MOVIES_TO_PROCESS:
        movie = movie_queue.get()
        try:
            print(f"Processing movie: {movie.title} (ID: {movie.movie_id})")
            get_movie_details(movie)
            append_movie_to_jsonl(movie, output_file)
            print(f"Successfully processed and saved '{movie.title}'.")
            processed_count += 1
        except Exception as e:
            print(f"Failed to process movie '{movie.title}' (ID: {movie.movie_id}). Error: {e}")
        
        print(f"{movie_queue.qsize()} movies remaining in the queue.")
        time.sleep(1)
