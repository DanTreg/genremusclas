# Initialize Kaggle API
import os
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm
import zipfile

# Ensure the Kaggle JSON file is in the correct location
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Dataset information
dataset = 'andradaolteanu/gtzan-dataset-music-genre-classification'
output_path = 'gtzan-dataset-music-genre-classification.zip'
extract_path = 'gtzan_genres'

# Function to download the dataset with progress bar
api.dataset_download_files(dataset, path='.', unzip=False)

# Extract the dataset
def extract_zip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc='Extracting '):
            zip_ref.extract(member, extract_path)
        print("Extraction complete")

# Extract the dataset
extract_zip_file(output_path, extract_path)