#!/usr/bin/env python
import os
import subprocess
import shutil

# Define the seeds and model keys
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

# Base directories
model_base_dir = "models"          # Models stored as models/seed-{seed}/{model_key}/{model_key}_step_100000.pt
opennmt_base_dir = "data/opennmt"    # Test files stored as data/opennmt/seed-{seed}/{folder}/src-test.txt and tgt-test.txt
predictions_base_dir = "predictions" # Output predictions will be saved here

# OpenNMT options
translate_options = ["-verbose", "-n_best", "3", "-beam_size", "3"]

def run_prediction(model_path, src_file, output_file):
    """Run the prediction command using onmt_translate."""
    command = ["onmt_translate",
               "-model", model_path,
               "-src", src_file,
               "-output", output_file] + translate_options
    print("Running command:", " ".join(command))
    subprocess.run(command, check=True)

def copy_test_files(src_file, tgt_file, dest_dir):
    """Copy the test source and target files into the given destination folder."""
    shutil.copy(src_file, os.path.join(dest_dir, os.path.basename(src_file)))
    shutil.copy(tgt_file, os.path.join(dest_dir, os.path.basename(tgt_file)))

def main():
    for seed in seeds:
        for model_key in model_keys:
            # Build the model file path
            model_path = os.path.join(model_base_dir, f"seed-{seed}", model_key, f"{model_key}_step_100000.pt")
            if not os.path.exists(model_path):
                print(f"Model file {model_path} not found. Skipping {model_key} for seed {seed}.")
                continue

            # Canonical Prediction Run
            # For canonical runs, always use the c1 test set for the given seed.
            canonical_src = os.path.join(opennmt_base_dir, f"seed-{seed}", "c1", "src-test.txt")
            canonical_tgt = os.path.join(opennmt_base_dir, f"seed-{seed}", "c1", "tgt-test.txt")
            canonical_out_dir = os.path.join(predictions_base_dir, f"seed-{seed}", f"{model_key}-canonical")
            os.makedirs(canonical_out_dir, exist_ok=True)
            canonical_pred = os.path.join(canonical_out_dir, "pred-test.txt")

            try:
                print(f"\n[Seed {seed}] Running canonical prediction for model '{model_key}' ...")
                run_prediction(model_path, canonical_src, canonical_pred)
                copy_test_files(canonical_src, canonical_tgt, canonical_out_dir)
                print(f"Canonical prediction completed for model '{model_key}' (seed {seed}).")
            except subprocess.CalledProcessError as e:
                print(f"Error during canonical prediction for {model_key} (seed {seed}): {e}")

            # Randomized Prediction Run
            # For randomized runs, use the nc1 test set for the given seed.
            randomized_src = os.path.join(opennmt_base_dir, f"seed-{seed}", "nc1", "src-test.txt")
            randomized_tgt = os.path.join(opennmt_base_dir, f"seed-{seed}", "nc1", "tgt-test.txt")
            randomized_out_dir = os.path.join(predictions_base_dir, f"seed-{seed}", f"{model_key}-randomized")
            os.makedirs(randomized_out_dir, exist_ok=True)
            randomized_pred = os.path.join(randomized_out_dir, "pred-test.txt")

            try:
                print(f"\n[Seed {seed}] Running randomized prediction for model '{model_key}' ...")
                run_prediction(model_path, randomized_src, randomized_pred)
                copy_test_files(randomized_src, randomized_tgt, randomized_out_dir)
                print(f"Randomized prediction completed for model '{model_key}' (seed {seed}).")
            except subprocess.CalledProcessError as e:
                print(f"Error during randomized prediction for {model_key} (seed {seed}): {e}")

if __name__ == "__main__":
    main()
