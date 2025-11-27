# from sentence_transformers import SentenceTransformer
# import numpy as np
# from typing import List, Dict
# from models import Movie, HUB_EMBEDDINGS_REPO_ID
# from datasets import load_dataset
# from scipy.spatial.distance import cosine
# import os
# from dotenv import load_dotenv
# from huggingface_hub import login
#
#
# def generate_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
#     """
#     Generates embeddings for a list of texts using the 'all-MiniLM-L12-v2' model.
#     The sentence-transformers library handles batching internally, so you can pass a large list of texts directly.
#
#     Args:
#         texts: A list of strings to be encoded.
#         batch_size: The batch size to use for the computation. A larger batch size can be faster but uses more VRAM.
#
#     Returns:
#         A numpy array containing the embeddings.
#     """
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
#     embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
#     return embeddings
#
# def load_movies_with_embeddings_from_hub() -> List[Movie]:
#     """
#     Loads the movie dataset with embeddings from Hugging Face Hub and converts each entry to a Movie object.
#
#     Returns:
#         A list of Movie objects, each containing its embedding.
#     """
#     print(f"Loading dataset from '{HUB_EMBEDDINGS_REPO_ID}'...")
#     dataset = load_dataset(HUB_EMBEDDINGS_REPO_ID, split="train")
#
#     movies = []
#     for item in dataset:
#         movie = Movie(
#             movie_id=item.get('movie_id'),
#             title=item.get('title'),
#             genres=item.get('genres', []),
#             plot_summary=item.get('plot_summary', ''),
#             director=item.get('director', ''),
#             stars=item.get('stars', []),
#             all_MiniLM_L12_v2_embedding=item.get('all_MiniLM_L12_v2_embedding', [])
#         )
#         # Only include movies that actually have an embedding
#         if movie.all_MiniLM_L12_v2_embedding:
#             movies.append(movie)
#
#     print(f"Successfully loaded and converted {len(movies)} movies with embeddings.")
#     return movies
#
# def get_recommendations(
#     query: str,
#     movies: List[Movie],
#     top_n: int = 5,
#     query_is_movie_title: bool = True
# ) -> List[Movie]:
#     """
#     Generates movie recommendations based on a query.
#
#     Args:
#         query: The movie title or a text string to base recommendations on.
#         movies: A list of Movie objects with pre-generated embeddings.
#         top_n: The number of top recommendations to return.
#         query_is_movie_title: If True, the query is treated as a movie title from the dataset.
#                               If False, the query is treated as a custom text string.
#
#     Returns:
#         A list of recommended Movie objects, sorted by similarity.
#     """
#     if not movies:
#         print("No movies loaded to generate recommendations from.")
#         return []
#
#     query_embedding = None
#     if query_is_movie_title:
#         # Find the movie in the dataset and use its embedding
#         query_movie = next((m for m in movies if m.title.lower() == query.lower()), None)
#         if query_movie and query_movie.all_MiniLM_L12_v2_embedding:
#             query_embedding = np.array(query_movie.all_MiniLM_L12_v2_embedding)
#             print(f"Generating recommendations based on movie: '{query_movie.title}'")
#         else:
#             print(f"Movie '{query}' not found or has no embedding. Trying to generate embedding from title and plot summary.")
#             # Fallback: generate embedding from the movie's string representation if found
#             if query_movie:
#                 query_embedding = generate_embeddings([query_movie.to_embedding_string()])[0]
#             else:
#                 print(f"Could not find movie '{query}'. Cannot generate recommendations.")
#                 return []
#     else:
#         # Generate embedding for the custom text query
#         print(f"Generating recommendations based on custom query: '{query}'")
#         query_embedding = generate_embeddings([query])[0]
#
#     if query_embedding is None:
#         return []
#
#     similarities = []
#     for movie in movies:
#         if movie.all_MiniLM_L12_v2_embedding:
#             movie_embedding = np.array(movie.all_MiniLM_L12_v2_embedding)
#             # Ensure embeddings are not zero vectors before calculating similarity
#             if np.all(movie_embedding == 0):
#                 continue
#
#             sim = 1 - cosine(query_embedding, movie_embedding)
#             similarities.append((sim, movie))
#
#     similarities.sort(key=lambda x: x[0], reverse=True)
#
#     # Filter out the query movie itself if it was a movie title
#     if query_is_movie_title and query_movie:
#         recommendations = [movie for sim, movie in similarities if movie.movie_id != query_movie.movie_id][:top_n]
#     else:
#         recommendations = [movie for sim, movie in similarities][:top_n]
#
#     return recommendations
#
#
# if __name__ == "__main__":
#     load_dotenv()
#
#     hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
#     if hf_token:
#         print("Logging in to Hugging Face Hub...")
#         login(token=hf_token)
#     else:
#         print("Warning: HUGGING_FACE_HUB_TOKEN not found. Cannot load dataset from Hub.")
#         exit()
#
#     all_movies = load_movies_with_embeddings_from_hub()
#
#     if all_movies:
#         while True:
#             user_input = input("\nEnter a movie title for recommendations (or type 'q' to quit, 't' for text query): ").strip()
#             if user_input.lower() == 'q':
#                 break
#
#             if user_input.lower() == 't':
#                 text_query = input("Enter a text query for recommendations: ").strip()
#                 if not text_query:
#                     print("Text query cannot be empty.")
#                     continue
#                 recommended_movies = get_recommendations(text_query, all_movies, top_n=5, query_is_movie_title=False)
#             else:
#                 recommended_movies = get_recommendations(user_input, all_movies, top_n=5, query_is_movie_title=True)
#
#             if recommended_movies:
#                 print("\n--- Top Recommendations ---")
#                 for i, movie in enumerate(recommended_movies):
#                     print(f"{i+1}. {movie.title} (Director: {movie.director}, Genres: {', '.join(movie.genres)})")
#             else:
#                 print("No recommendations found.")
#     else:
#         print("Could not load movies with embeddings. Exiting.")
