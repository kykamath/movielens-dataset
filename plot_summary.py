import torch
from transformers import pipeline

# 1. Define the LLM to use (a small, capable model)
# We use a Causal LM, which is designed for generation and answering questions.
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# 2. Set up the Instruction/Prompt
# This is the "title-only" input and the instruction for the summary.
movie_title = "The Matrix"

prompt = f"""
[INST] You are an expert film critic and plot summarizer. Given only a movie title, generate a brief, but comprehensive plot summary (100-150 words). Do not include any cast, crew, or release date information.

Movie Title: {movie_title}
Plot Summary:
[/INST]
"""

# 3. Create the Text Generation Pipeline
# Using the pipeline simplifies model loading and generation parameters.
# We set trust_remote_code=True if needed for some models.
generator = pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.bfloat16,  # Use a more efficient data type if supported by your hardware
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

# 4. Generate the Summary
result = generator(
    prompt,
    max_new_tokens=150,    # Limit the length of the generated summary
    do_sample=True,        # Use sampling for more creative/natural text
    temperature=0.7,       # Control randomness (0.7 is a good balance)
    top_p=0.95,            # Control diversity
    pad_token_id=generator.tokenizer.eos_token_id
)

# 5. Extract and Print the Cleaned Summary
# The output will contain the full prompt, so we extract the generated text.
generated_text = result[0]['generated_text'].replace(prompt, '').strip()

print(f"--- Summary for: {movie_title} ---")
print(generated_text)
print("---------------------------------")