from huggingface_hub import snapshot_download
import json
import os

# Step 1: Download the model locally
model_dir = snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# Step 2: Modify rope_scaling config if needed
config_path = os.path.join(model_dir, "config.json")
with open(config_path, "r+") as f:
    config = json.load(f)
    rope_scaling = config.get("rope_scaling", None)
    if rope_scaling and "type" not in rope_scaling:
        config["rope_scaling"]["type"] = "yarn"  # or "linear", if that's the intended one
        f.seek(0)
        json.dump(config, f, indent=2)
        f.truncate()

print("✅ Model config patched successfully.")
print("Model Dir:",model_dir)