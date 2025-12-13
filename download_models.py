import os
from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm

# put download model_id here
MODEL_IDS = [

]

# local directory to save models
BASE_DIR = "./models" 

api = HfApi()

def download_models(model_ids, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    for model_id in model_ids:
        folder = model_id.replace("/", "_")
        dest = os.path.join(base_dir, folder)
        if os.path.isdir(dest):
            print(f"[Skipped] {model_id} already exists ({dest})")
            continue

        print(f"\n[Starting] {model_id} → {dest}")
        os.makedirs(dest, exist_ok=True)

        try:
            files = api.list_repo_files(repo_id=model_id)
        except Exception as e:
            print(f"[Error] Unable to list files for {model_id}: {e}")
            continue

        for fname in tqdm(files, desc=f"Downloading files for {model_id}", unit="file"):
            try:
                hf_hub_download(
                    repo_id=model_id,
                    filename=fname,
                    local_dir=dest,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
            except Exception as e:
                print(f"    ❌ download fail {fname}: {e}")

        print(f"[Completed] {model_id}")

if __name__ == "__main__":
    download_models(MODEL_IDS, BASE_DIR)