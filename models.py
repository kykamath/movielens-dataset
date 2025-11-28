from dataclasses import dataclass, field
from typing import List

# --- Hugging Face Hub Repository IDs ---
HUB_ENRICHED_REPO_ID = "krishnakamath/movielens-32m-movies-enriched"

# --- Local File Paths ---
ENRICHED_MOVIES_JSONL = "movies_with_details.jsonl"

@dataclass
class Movie:
    movie_id: int
    title: str
    genres: List[str]
    plot_summary: str = ""
    director: str = ""
    stars: List[str] = field(default_factory=list)
