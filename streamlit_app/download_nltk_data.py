import os
import nltk

# Get chatbot2025 folder (one level up from streamlit_app)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLTK_DATA_PATH = os.path.join(BASE_DIR, 'nltk_data')

print(f"Downloading NLTK data to: {NLTK_DATA_PATH}")

nltk.download("punkt", download_dir=NLTK_DATA_PATH)
nltk.download("wordnet", download_dir=NLTK_DATA_PATH)

print("Download complete!")