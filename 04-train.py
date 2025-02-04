import os
import subprocess

augmentations = [
    'not_augmented',
    'augmented_2x',
    'augmented_5x',
    'augmented_10x',
    'augmented_20x',
    'augmented_50x',
    'mixed_augmented',
    'partial_augmented_5x',
    'not_augmented_scrambled',
    'partial_augmented_5x_scrambled',
]

for augmentation in augmentations:
    # Change directory
    os.chdir(f'/home/markus/Developer/Code/NPstereo/models/{augmentation}/')
    print(f"Current working directory: {os.getcwd()}")
    
    # Build vocabulary
    print("Building vocabulary...")
    subprocess.run(['onmt_build_vocab', '-config', 'config.yaml'], check=True)

    # Train model
    print("Training model...")
    subprocess.run(['onmt_train', '-config', 'config.yaml'], check=True)

print("All augmentations processed.")
