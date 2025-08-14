import os
import subprocess
import time
import torch
import glob
import gc

# List of seeds and model keys from config generation.
seeds = [0, 1, 42]
model_keys = [
    "c1",           # not augmented
    "r1",           # not augmented scrambled
    "a2",           # augmented 2x
    "a5",           # augmented 5x
    "a10",          # augmented 10x
    "a20",          # augmented 20x
    "a50",          # augmented 50x
    "npstereo",     # partial augmented 5x
    "rp",           # scrambled partial augmented 5x
    "m65"           # mixed augmented
]

# Base directory where models are stored.
base_model_dir = "/xxx/NPstereo/models" # Adjust this path as necessary

# Copy current environment and set PYTORCH_CUDA_ALLOC_CONF to help with memory fragmentation.
env = os.environ.copy()
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

max_retries = 3

for seed in seeds:
    for model_key in model_keys:
        model_dir = os.path.join(base_model_dir, f"seed-{seed}", model_key)
        if not os.path.exists(model_dir):
            print(f"Skipping non-existent directory: {model_dir}")
            continue

        # Skip if training has already been done (.pt files exist).
        pt_files = glob.glob(os.path.join(model_dir, "*.pt"))
        if pt_files:
            print(f"Model already trained in {model_dir} (found {len(pt_files)} .pt file(s)). Skipping training.")
            continue

        os.chdir(model_dir)
        print(f"\nProcessing model '{model_key}' for seed {seed} in {os.getcwd()}")

        # Build vocabulary using OpenNMT's tool.
        print("Building vocabulary...")
        subprocess.run(['onmt_build_vocab', '-config', 'config.yaml'], env=env, check=True)

        # Train the model using OpenNMT with a retry mechanism.
        retry_count = 0
        while retry_count < max_retries:
            print(f"Training model (attempt {retry_count + 1}/{max_retries})...")
            try:
                subprocess.run(['onmt_train', '-config', 'config.yaml'], env=env, check=True)
                print("Training completed successfully.")
                break
            except subprocess.CalledProcessError as e:
                error_str = str(e)
                if ("No inf checks were recorded for this optimizer" in error_str) or ("cuda OOM" in error_str):
                    print(f"Encountered an AMP/cuda OOM error for model '{model_key}' with seed {seed}.")
                    print("Clearing cache and retrying in 30 seconds...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(30)
                    retry_count += 1
                else:
                    print(f"Training failed for model '{model_key}' with seed {seed}. Error: {e}")
                    break

        if retry_count == max_retries:
            print(f"Max retries reached for model '{model_key}' with seed {seed}. Skipping training for this model.")
            continue

        # Clear GPU cache and perform garbage collection before moving on.
        torch.cuda.empty_cache()
        gc.collect()

        # Optional delay between training runs.
        time.sleep(10)

print("All models processed.")