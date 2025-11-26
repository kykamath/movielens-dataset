# Enriched MovieLens 32M Dataset

## Dataset Description

This dataset is an enriched version of the popular [MovieLens 32M dataset](https://grouplens.org/datasets/movielens/). The original `movies.csv` file, which contains `movieId`, `title`, and `genres`, has been augmented with additional details for each movie. These details were generated using OpenAI's `gpt-3.5-turbo` model.

The added fields include:
-   **Plot Summary**: A brief but comprehensive summary of the movie's plot (100-150 words).
-   **Director**: The name of the movie's director.
-   **Stars**: A list of the top 3 starring actors in the movie.

This enriched dataset is ideal for tasks that require more than just basic movie metadata, such as content-based recommendation systems, natural language processing tasks on plot summaries, or creating more detailed movie information displays.

## Dataset Structure

The enriched dataset is available in [krishnakamath/movielens-32m-enriched](https://huggingface.co/krishnakamath/movielens-32m-enriched).

### Data Fields

-   `movie_id` (integer): A unique identifier for each movie.
-   `title` (string): The title of the movie, including the year of release.
-   `genres` (list of strings): A list of genres associated with the movie.
-   `plot_summary` (string): A summary of the movie's plot.
-   `director` (string): The director of the movie.
-   `stars` (list of strings): A list containing the names of the top 3 actors.

### Example

```json
{
  "movie_id": 1,
  "title": "Toy Story (1995)",
  "genres": ["Adventure", "Animation", "Children", "Comedy", "Fantasy"],
  "plot_summary": "In a world where toys are living things who pretend to be lifeless when humans are present, a group of toys are caught off guard when their owner Andy gets a new toy, a Buzz Lightyear action figure. The previous favorite toy, a cowboy doll named Woody, is now jealous of Buzz and the two must learn to put aside their differences when they are separated from their owner and must find their way back home.",
  "director": "John Lasseter",
  "stars": ["Tom Hanks", "Tim Allen", "Don Rickles"]
}
```

## How to Use

You can easily load this dataset using the Hugging Face `datasets` library.

```python
from datasets import load_dataset

# Load the dataset from the Hugging Face Hub
# Replace with your actual Hub repository ID
dataset = load_dataset("krishnakamath/movielens-32m-enriched")

# Access the data
print(dataset['train'][0])
```

## Dataset Creation

The dataset was generated using the `create_dataset.py` script in this repository. The script performs the following steps:
1.  Reads the `movies.csv` from the MovieLens 32M dataset.
2.  For each movie, it calls the OpenAI API (`gpt-3.5-turbo`) to generate the plot summary, director, and stars.
3.  It saves the enriched data into the `movies_with_details.jsonl` file.
4.  The script includes logic to resume processing and avoid re-enriching movies that are already present in the output file.

## Citation

### Original Dataset

Please cite the original MovieLens dataset if you use this data in your research:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>

### Enrichment Process

The enrichment of this dataset was performed using the OpenAI API. Please acknowledge the use of their models if this dataset is beneficial to your work.

## Acknowledgement

The Python scripts used to generate and process this dataset were developed with the assistance of Google's Gemini.
