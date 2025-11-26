import pandas as pd
from datasets import Dataset
from huggingface_hub import login
import os
import json
from openai import OpenAI
from dotenv import load_dotenv


def get_movie_details(movie_title: str) -> dict:
    """
    Generates a plot summary, director, and top 3 stars for a given movie title using the OpenAI API.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = f"""
    [INST] You are an expert film critic. Given a movie title, provide a JSON object with the following fields:
    - "plot_summary": A brief, but comprehensive plot summary (100-150 words).
    - "director": The name of the director.
    - "stars": A list of the top 3 starring actors.

    Do not include any other information.

    Movie Title: {movie_title}
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
    return details_json

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
    movie_title = "Toy Story (1995)"
    details = get_movie_details(movie_title)
    print(f"Details for '{movie_title}':")
    print(f"  Plot Summary: {details.get('plot_summary')}")
    print(f"  Director: {details.get('director')}")

    stars = details.get('stars', [])
    if stars:
        print(f"  Stars: {', '.join(stars)}")

# if __name__ == "__main__":
#     login()
#     HUB_DATASET_REPO_ID = "krishnakamath/movielens-32m-movies"
#
#     # Load and display the dataset
#     dataset = load_movies_dataset("ml-32m/movies.csv")
#     print(f"Dataset loaded with {len(dataset)} movies")
#     print(f"Columns: {dataset.column_names}")
#     print("\nFirst few rows:")
#     print(dataset.to_pandas().head())
#
#     # Push to Hugging Face Hub (uncomment and set repo_id to use)
#     push_dataset_to_hub(dataset, repo_id=HUB_DATASET_REPO_ID, private=False)