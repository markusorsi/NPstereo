import os

# List of seeds and model keys (names).
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

# Base directory where models will be saved.
base_model_dir = "/xxx/NPstereo/models" # Adjust this path as necessary
base_opennmt_dir = "../../../data/opennmt"

# Template for the config file.
config_template = """\
## Where the samples will be written
save_data: models/seed-{seed}/{model_key}/

## Where the vocab(s) will be written
src_vocab: {model_key}.vocab.src
tgt_vocab: {model_key}.vocab.tgt

# Prevent overwriting existing files in the folder
overwrite: True

# Tensorboard 
tensorboard: true
tensorboard_log_dir: logs

# Corpus opts:
data:
    train:
        path_src: {opennmt_dir}/seed-{seed}/{model_key}/src-train.txt
        path_tgt: {opennmt_dir}/seed-{seed}/{model_key}/tgt-train.txt
    valid:
        path_src: {opennmt_dir}/seed-{seed}/{model_key}/src-val.txt
        path_tgt: {opennmt_dir}/seed-{seed}/{model_key}/tgt-val.txt

# Vocabulary files that were just created
src_vocab: {model_key}.vocab.src
tgt_vocab: {model_key}.vocab.tgt

# General opts
save_model: {model_key}
save_checkpoint_steps: 100000
train_steps: 100000
valid_steps: 5000

# Batching
bucket_size: 262144
world_size: 1
gpu_ranks: [0]
num_workers: 4
batch_type: "tokens"
batch_size: 4096
valid_batch_size: 2048
accum_count: [4]
accum_steps: [0]

# Optimization
model_dtype: "fp16"
optim: "adam"
learning_rate: 2
warmup_steps: 8000
decay_method: "noam"
adam_beta2: 0.998
max_grad_norm: 0
label_smoothing: 0.1
param_init: 0
param_init_glorot: true
normalization: "tokens"

# Model
encoder_type: transformer
decoder_type: transformer
position_encoding: true
enc_layers: 6
dec_layers: 6
heads: 8
hidden_size: 512
word_vec_size: 512
transformer_ff: 2048
dropout_steps: [0]
dropout: [0.1]
attention_dropout: [0.1]
"""

# Iterate over each seed and model key to create directories and config files.
for seed in seeds:
    for model_key in model_keys:
        # Define the model directory (e.g. models/seed-1/a2)
        model_dir = os.path.join(base_model_dir, f"seed-{seed}", model_key)
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate the config file content with the proper values.
        config_content = config_template.format(
            seed=seed,
            model_key=model_key,
            opennmt_dir=base_opennmt_dir
        )
        
        # Write the config file.
        config_path = os.path.join(model_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(config_content)
        
        print(f"Generated config for model '{model_key}' (seed {seed}) at: {config_path}")
