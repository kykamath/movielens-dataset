# import os
# import json
# from openai import OpenAI
# from dotenv import load_dotenv
#
#
# def get_movie_details(movie_title: str) -> dict:
#     """
#     Generates a plot summary, director, and top 3 stars for a given movie title using the OpenAI API.
#     """
#     client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#
#     prompt = f"""
#     [INST] You are an expert film critic. Given a movie title, provide a JSON object with the following fields:
#     - "plot_summary": A brief, but comprehensive plot summary (100-150 words).
#     - "director": The name of the director.
#     - "stars": A list of the top 3 starring actors.
#
#     Do not include any other information.
#
#     Movie Title: {movie_title}
#     [/INST]
#     """
#
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are an expert in movies."},
#             {"role": "user", "content": prompt},
#         ],
#         max_tokens=400,
#         temperature=0.7,
#         top_p=0.95,
#         response_format={"type": "json_object"},
#     )
#
#     details_json = json.loads(response.choices[0].message.content)
#     return details_json
#
#
# if __name__ == "__main__":
#     load_dotenv()
#     movie_title = "Toy Story (1995)"
#     details = get_movie_details(movie_title)
#     print(f"Details for '{movie_title}':")
#     print(f"  Plot Summary: {details.get('plot_summary')}")
#     print(f"  Director: {details.get('director')}")
#
#     stars = details.get('stars', [])
#     if stars:
#         print(f"  Stars: {', '.join(stars)}")
#
