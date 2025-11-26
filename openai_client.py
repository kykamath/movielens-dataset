import os
from openai import OpenAI
from dotenv import load_dotenv


def get_plot_summary(movie_title: str) -> str:
    """
    Generates a plot summary for a given movie title using the OpenAI API.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = f"""
    [INST] You are an expert film critic and plot summarizer. Given only a movie title, generate a brief, but comprehensive plot summary (100-150 words). Do not include any cast, crew, or release date information.
    
    Movie Title: {movie_title}
    Plot Summary:
    [/INST]
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.7,
        top_p=0.95,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    load_dotenv()
    movie_title = "The Matrix"
    summary = get_plot_summary(movie_title)
    print(f"Plot summary for '{movie_title}':")
    print(summary)
