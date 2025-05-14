#!/usr/bin/env python
"""
Download sample audio files from the RAVDESS dataset for speech sentiment analysis.
"""
import os
import requests
import zipfile
import tempfile
import shutil
from tqdm import tqdm

def download_file(url, target_path, desc=None):
    """Download a file from a URL to a target path."""
    if os.path.exists(target_path):
        print(f"File already exists at {target_path}")
        return
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Get file size for progress bar
    file_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    with open(target_path, 'wb') as f, tqdm(
            desc=desc or os.path.basename(target_path),
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)
    
    print(f"Downloaded {url} to {target_path}")
    return target_path

def main():
    """Download sample files for the project."""
    # RAVDESS dataset zip file URL
    zip_url = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    
    # Create temp directory for the zip file
    temp_dir = tempfile.mkdtemp()
    temp_zip_path = os.path.join(temp_dir, "Audio_Speech_Actors_01-24.zip")
    
    # Target files we want to extract
    target_files = [
        # Neutral samples
        "Actor_01/03-01-01-01-01-01-01.wav",  # Neutral, female
        "Actor_11/03-01-01-01-01-01-11.wav",  # Neutral, male
        
        # Happy samples
        "Actor_01/03-01-03-01-01-01-01.wav",  # Happy, female
        "Actor_11/03-01-03-01-01-01-11.wav",  # Happy, male
        
        # Sad samples
        "Actor_01/03-01-04-01-01-01-01.wav",  # Sad, female
        "Actor_11/03-01-04-01-01-01-11.wav",  # Sad, male
        
        # Angry samples
        "Actor_01/03-01-05-01-01-01-01.wav",  # Angry, female
        "Actor_11/03-01-05-01-01-01-11.wav"   # Angry, male
    ]
    
    # Directory to save audio files
    audio_dir = os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(audio_dir, exist_ok=True)
    
    try:
        # Download the zip file
        print(f"Downloading RAVDESS Speech dataset zip file...")
        download_file(zip_url, temp_zip_path, desc="RAVDESS Speech dataset")
        
        # Extract specific files
        print(f"Extracting {len(target_files)} audio samples...")
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            # Get all file paths in the zip (to check path structure)
            all_files = zip_ref.namelist()
            
            # Extract only the target files
            for file_path in target_files:
                # Need to include the parent directory ("Audio_Speech_Actors_01-24")
                full_path_in_zip = None
                for path in all_files:
                    if path.endswith(file_path):
                        full_path_in_zip = path
                        break
                
                if full_path_in_zip:
                    # Get just the filename without path
                    filename = os.path.basename(file_path)
                    
                    # Extract and save to the samples directory
                    target_path = os.path.join(audio_dir, filename)
                    
                    with zip_ref.open(full_path_in_zip) as source, open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    
                    print(f"Extracted: {filename}")
                else:
                    print(f"Warning: Could not find {file_path} in zip file")
        
        print(f"Extracted audio files to {audio_dir}")
        print("These can be used for speech-to-text sentiment analysis using Whisper.")
    
    finally:
        # Clean up the temp directory
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main() 