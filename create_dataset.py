import pandas as pd
from datasets import Dataset

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
    HUB_DATASET_REPO_ID = "krishnakamath/movielens-32m-movies"

    # Load and display the dataset
    dataset = load_movies_dataset("ml-32m/movies.csv")
    print(f"Dataset loaded with {len(dataset)} movies")
    print(f"Columns: {dataset.column_names}")
    print("\nFirst few rows:")
    print(dataset.to_pandas().head())
    
    # Push to Hugging Face Hub (uncomment and set repo_id to use)
    push_dataset_to_hub(dataset, repo_id=HUB_DATASET_REPO_ID, private=False)