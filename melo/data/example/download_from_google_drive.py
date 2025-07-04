import os
import subprocess
import re
import requests

# Set your folder ID and output directory
FOLDER_ID = "1TTOkoO1TWbwWGDaoIWOobRL0pILbTr4r"
OUTPUT_DIR = "./wavs"

def get_file_list_from_drive_folder(folder_id):
    """
    Uses the undocumented Google Drive folder HTML page to extract file IDs and names.
    This only works for public folders.
    """
    print("Fetching file list from Drive folder...")
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    response = requests.get(url)
    if "drive_folder" not in response.text:
        raise Exception("Cannot access folder or folder is not public.")

    # Extract file IDs and names using regex (basic, but works for public folders)
    pattern = r'"(https://drive.google.com/file/d/[^/]+/view\?usp=drive_link)"'
    links = list(set(re.findall(pattern, response.text)))
    file_ids = [re.search(r"/file/d/([^/]+)/", link).group(1) for link in links]
    print(f"Found {len(file_ids)} files.")
    return file_ids

def download_files(file_ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, file_id in enumerate(file_ids, 1):
        print(f"[{i}/{len(file_ids)}] Downloading file {file_id}")
        subprocess.run([
            "gdown",
            "--id", file_id,
            "--output", os.path.join(output_dir, f"{file_id}.file")
        ])

if __name__ == "__main__":
    ids = get_file_list_from_drive_folder(FOLDER_ID)
    download_files(ids, OUTPUT_DIR)

