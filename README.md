# Bridge Bidding RL — Transformer Architecture

This repository is a fork of [harukaki/brl](https://github.com/harukaki/brl), extending the bridge bidding reinforcement learning framework from Kita et al. (2024) with a **Transformer encoder architecture** as an alternative to the original MLP.

## What This Fork Adds

- `BridgeTransformer` — a 3-layer Transformer encoder (d_model=256, 8 heads, 2.47M params) in `src/models.py`, replacing the flat MLP for the policy/value network
- Updated `sl.py` to support `type_of_model=Transformer` for supervised learning warm-start
- Updated `ppo.py` to support `actor_model_type=Transformer` for RL training
- Compatibility fixes for `pgx==2.4.0` and `jax==0.4.35` (newer versions than the original repo)
- Pre-trained SL warm-start models: `sl_model_deepmind/model_final.pkl` and `sl_model_transformer/model_final.pkl`

## Architecture Overview

The `BridgeTransformer` treats the 480-dimensional observation as two components:
- **Sequential:** the 420-bit bidding history is reshaped into 35 tokens × 12 features, processed by a Transformer encoder with learned positional embeddings
- **Global:** the 52-bit hand and 8-bit context (vulnerability) are concatenated after pooling

This is motivated by the fact that bridge bidding is inherently relational — the meaning of any bid depends on its relationship to all other bids in the auction, which self-attention captures more expressively than fixed MLP weights.

---

## Installation

> ⚠️ **Do not run `pip install -r requirements.txt` directly.** JAX requires a separate installation with CUDA support. Follow the steps below exactly.

### Step 1 — Install JAX with GPU support

```bash
pip uninstall jax jaxlib jax-cuda12-plugin -y
pip install "jax[cuda12]==0.4.35" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Step 2 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Verify GPU is detected

```python
import jax
print(jax.__version__)   # should be 0.4.35
print(jax.devices())     # should show CudaDevice, not CpuDevice
```

### Step 4 — Download DDS dataset

```python
from pgx.bridge_bidding import download_dds_results
download_dds_results()
```

Then create the required symlinks:

```bash
ln -sf $(pwd)/dds_results/dds_results_500K.npy dds_results/test_000.npy
ln -sf $(pwd)/dds_results/dds_results_10M.npy dds_results/dds_results_train_000.npy
ln -sf $(pwd)/dds_results/dds_results_2.5M.npy dds_results/dds_results_train_001.npy
```

### Step 5 — Download supervised learning data

```bash
mkdir -p bridge_data
gsutil -m cp gs://openspiel-data/bridge/train.txt bridge_data/
gsutil -m cp gs://openspiel-data/bridge/test.txt bridge_data/
```

---

## Pre-trained SL Warm-Start Models

This fork includes pre-trained supervised learning warm-start models trained on the OpenSpiel bridge dataset. These are required before running RL training.

| File | Architecture | Purpose |
|------|-------------|---------|
| `sl_model_transformer/model_final.pkl` | BridgeTransformer (2.47M params) | Warm-start for Transformer RL training |
| `sl_model_deepmind/model_final.pkl` | DeepMind MLP (3.7M params) | Warm-start for MLP RL training + opponent model |

> **Note:** These models were trained with `jax==0.4.35` and `dm-haiku==0.0.16`. They are **not compatible** with the original Kita et al. pkl files which were saved with older JAX versions.

---

## Training

### Recommended parameters

The following parameters are calibrated for a single A100/H100 GPU (~2-3 hours) while producing meaningful results for architectural comparison:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `num_envs` | 1024 | 8x fewer than paper's 8192; still sufficient parallelism |
| `num_steps` | 32 | Same as paper |
| `total_timesteps` | 200000000 | 10% of paper; enough to show clear learning trends |
| `update_epochs` | 10 | Same as paper |
| `minibatch_size` | 512 | Halved to match reduced num_envs |
| `lr` | 0.000001 | Same as paper |

### Train Transformer (main experiment)

```bash
mkdir -p rl_log/transformer_full/rl_params

XLA_PYTHON_CLIENT_PREALLOCATE=false python ppo.py \
  actor_model_type=Transformer \
  self_play=False \
  opp_model_type=DeepMind \
  opp_model_path="sl_model_deepmind/model_final.pkl" \
  num_envs=1024 \
  num_steps=32 \
  total_timesteps=200000000 \
  update_epochs=10 \
  minibatch_size=512 \
  lr=0.000001 \
  gamma=1 \
  gae_lambda=0.95 \
  ent_coef=0.001 \
  VE_COEF=0.5 \
  load_initial_model=True \
  initial_model_path="sl_model_transformer/model_final.pkl" \
  eval_opp_model_path="sl_model_deepmind/model_final.pkl" \
  eval_opp_model_type=DeepMind \
  num_eval_envs=100 \
  num_eval_step=10 \
  save_model=True \
  save_model_interval=50 \
  track=False \
  log_path="rl_log" \
  exp_name="transformer_full"
```

### Train DeepMind MLP baseline (for comparison)

```bash
mkdir -p rl_log/mlp_full/rl_params

XLA_PYTHON_CLIENT_PREALLOCATE=false python ppo.py \
  actor_model_type=DeepMind \
  self_play=False \
  opp_model_type=DeepMind \
  opp_model_path="sl_model_deepmind/model_final.pkl" \
  num_envs=1024 \
  num_steps=32 \
  total_timesteps=200000000 \
  update_epochs=10 \
  minibatch_size=512 \
  lr=0.000001 \
  gamma=1 \
  gae_lambda=0.95 \
  ent_coef=0.001 \
  VE_COEF=0.5 \
  load_initial_model=True \
  initial_model_path="sl_model_deepmind/model_final.pkl" \
  eval_opp_model_path="sl_model_deepmind/model_final.pkl" \
  eval_opp_model_type=DeepMind \
  num_eval_envs=100 \
  num_eval_step=10 \
  save_model=True \
  save_model_interval=50 \
  track=False \
  log_path="rl_log" \
  exp_name="mlp_full"
```

> **Tip for long runs:** use `tmux` so training survives SSH disconnection:
> ```bash
> tmux new -s training
> # run training command
> # detach with Ctrl+B then D
> # reconnect with: tmux attach -t training
> ```

---

## Supervised Learning (optional — pre-trained models already included)

Only needed if you want to retrain the SL warm-starts from scratch.

### Train SL warm-start for Transformer

```bash
python sl.py \
  iterations=100000 \
  train_batch=128 \
  learning_rate=0.0001 \
  eval_every=10000 \
  data_path="bridge_data" \
  save_path="sl_model_transformer" \
  type_of_model=Transformer \
  track=False
```

### Train SL warm-start for DeepMind MLP

```bash
python sl.py \
  iterations=100000 \
  train_batch=128 \
  learning_rate=0.0001 \
  eval_every=10000 \
  data_path="bridge_data" \
  save_path="sl_model_deepmind" \
  type_of_model=DeepMind \
  track=False
```

---

## Evaluation

Compare two trained models head-to-head:

```bash
python eval.py \
  team1_model_path="rl_log/transformer_full/rl_params/params-XXXXX.pkl" \
  team2_model_path="rl_log/mlp_full/rl_params/params-XXXXX.pkl" \
  team1_model_type=Transformer \
  team2_model_type=DeepMind \
  num_eval_envs=1000
```

Replace `params-XXXXX.pkl` with the latest checkpoint filename from each experiment folder.

---

## Google Colab Setup

If running on Colab, add this before the installation steps:

```python
# Set runtime to GPU first: Runtime → Change runtime type → T4 GPU

# Cell 0 — Install (run once, then restart runtime)
!pip uninstall jax jaxlib jax-cuda12-plugin -y --quiet
!pip install "jax[cuda12]==0.4.35" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --quiet
!pip install -r requirements.txt --quiet
print("Done — restart runtime now")
```

After restarting, run the DDS download and symlink steps, then proceed with training commands above using `!` prefix for shell commands.

---

## Repository Structure

```
brl/
├── src/
│   ├── models.py        # BridgeTransformer + ActorCritic (MLP) definitions
│   ├── evaluation.py    # Evaluation functions
│   ├── roll_out.py      # PPO rollout
│   ├── update.py        # PPO update step
│   ├── gae.py           # Generalised Advantage Estimation
│   ├── duplicate.py     # Duplicate bridge scoring
│   └── utils.py         # Environment wrappers
├── sl.py                # Supervised learning training
├── ppo.py               # PPO reinforcement learning training
├── eval.py              # Model evaluation
├── sl_model_deepmind/   # Pre-trained SL warm-start (DeepMind MLP)
├── sl_model_transformer/ # Pre-trained SL warm-start (Transformer)
├── requirements.txt
└── README.md
```

---

## Key Differences from Original Repo

| | Original (harukaki/brl) | This Fork |
|---|---|---|
| Architecture | DeepMind MLP / FAIR MLP | + BridgeTransformer encoder |
| JAX version | 0.4.23 | 0.4.35 |
| pgx version | 1.4.0 | 2.4.0 |
| SL training | Returns logits only | Returns (logits, value) via make_forward_pass |
| Evaluation | Both teams use actor model type | Teams use their respective model types |
| utils.py | Uses state._rng_key (removed in pgx 2.x) | Passes rng explicitly |

---

## License

Apache 2.0

## Citations

```bibtex
@inproceedings{Kita2024,
  title  = {{A Simple, Solid, and Reproducible Baseline for Bridge Bidding AI}},
  author = {Kita, Haruka and Koyamada, Sotetsu and Yamaguchi, Yotaro and Ishii, Shin},
  year   = 2024,
  booktitle = {IEEE Conference on Games},
}

@inproceedings{Gong2019,
  title  = {Simple is Better: Training an End-to-end Contract Bridge Bidding Agent without Human Knowledge},
  author = {Gong, Qucheng and Jiang, Tina and Tian, Yuandong},
  year   = 2019,
  booktitle = {Real-world Sequential Decision Making Workshop at ICML},
}
```
