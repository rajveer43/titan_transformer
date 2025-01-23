from huggingface_hub import snapshot_download 

repo_id = "rajveer43/titan-transformer"

local_dir = "/home/cimcon/Documents/projects/titans"

snapshot_download(repo_id, local_dir=local_dir)

print(f"Repository downloaded to {local_dir}")
