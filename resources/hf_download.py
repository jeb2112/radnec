# quick script to download the base SAM from huggingface to local cache

from huggingface_hub import snapshot_download
from transformers import SamModel

if False: # pulls extra files which aren't needed just for inference
    snapshot_download(repo_id = "facebook/sam-vit-base")
else:
    m = SamModel.from_pretrained(f"facebook/sam-vit-base")