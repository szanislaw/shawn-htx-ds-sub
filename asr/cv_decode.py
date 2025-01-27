import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Filepath definitions
AUDIO_FOLDER = "common-voice/cv-valid-dev/"  # Path to the folder containing audio files
INPUT_CSV = "common-voice/cv-valid-dev.csv"  # Input CSV file path
OUTPUT_CSV = "common-voice/cv-valid-dev.csv"  # Output CSV file path

# Function to call the ASR API
def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            response = requests.post("http://localhost:8001/asr", files={"file": audio_file})
            if response.status_code == 200:
                result = response.json()
                # Convert transcription to lowercase
                return result.get("transcription", "").lower()
            else:
                logging.error(f"Failed to transcribe {file_path}. HTTP {response.status_code}: {response.text}")
                return None
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None

# Function to process each row in the DataFrame
def process_row(row):
    audio_file_path = Path(AUDIO_FOLDER) / row["filename"]
    if audio_file_path.exists():
        return transcribe_audio(audio_file_path)
    else:
        logging.warning(f"File not found: {audio_file_path}")
        return None

# Main function to handle the transcription process
def main():
    # Load the input CSV file
    logging.info(f"Loading existing {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Add the new column 'generated_text' if not already present
    if "generated_text" not in df.columns:
        df["generated_text"] = None

    # Create a ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit tasks for all rows in the DataFrame
        futures = {executor.submit(process_row, row): idx for idx, row in df.iterrows()}

        # Process completed tasks as they finish
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[future]
            transcription = future.result()
            df.at[idx, "generated_text"] = transcription

    # Save the updated CSV file
    logging.info(f"Saving updated CSV file to {OUTPUT_CSV}...")
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Transcriptions saved to {OUTPUT_CSV}.")

if __name__ == "__main__":
    main()