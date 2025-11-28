import os
from dotenv import load_dotenv
from huggingface_hub import login

from models import HUB_ENRICHED_REPO_ID, ENRICHED_MOVIES_JSONL
from enrichment_utils import upload_enriched_dataset

def main():
    """
    Uploads the local movies_with_details.jsonl file to the Hugging Face Hub.
    """
    # --- Configuration ---
    LOCAL_FILE = ENRICHED_MOVIES_JSONL

    # --- 1. Authentication ---
    load_dotenv()
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    can_upload = False
    if hf_token:
        print("Logging in to Hugging Face Hub...")
        login(token=hf_token)
        can_upload = True
    else:
        print("Warning: HUGGING_FACE_HUB_TOKEN not found. Upload to Hub will be skipped.")
        return

    # --- 2. Check for local file ---
    if not os.path.exists(LOCAL_FILE):
        print(f"Error: Local file '{LOCAL_FILE}' not found. Nothing to upload.")
        return

    # --- 3. Upload Dataset to Hugging Face Hub ---
    if can_upload:
        print(f"Found local file '{LOCAL_FILE}'. Starting upload...")
        upload_enriched_dataset(LOCAL_FILE, repo_id=HUB_ENRICHED_REPO_ID, private=False)
    else:
        # This case is already handled above, but included for completeness
        print("Skipping upload to Hugging Face Hub as login could not be completed.")

if __name__ == "__main__":
    main()
