# =========================
# Global Configuration
# =========================

# -------------------------
# Environment
# -------------------------
ENV_ID = "BreakoutNoFrameskip-v4"
MAX_STEPS = 1000

# -------------------------
# Data
# -------------------------
DATA_DIR = "data"

# -------------------------
# VAE
# -------------------------
LATENT_DIM = 32
FRAME_STACK = 4
FRAME_SIZE = 84

VAE_EPOCHS = 20
VAE_LR = 1e-3
VAE_SAVE_DIR = "artifacts/vae"
VAE_PATH = "artifacts/vae/vae_final.pt"

# -------------------------
# RNN (World Model)
# -------------------------
RNN_HIDDEN_DIM = 256
N_MIXTURES = 5
SEQ_LEN = 32
RNN_EPOCHS = 20
RNN_LR = 1e-3
REWARD_COEF = 1.0

RNN_SAVE_DIR = "artifacts/rnn"
RNN_PATH = "artifacts/rnn/rnn_best.pt"

# -------------------------
# Controller (Random Search / ES)
# -------------------------
ACTION_DIM = 4

POPULATION_SIZE = 30
ELITE_FRACTION = 0.2
NOISE_STD = 0.1
CONTROLLER_ITERS = 25
EPISODES_PER_EVAL = 2

WORLD_TEMPERATURE = 1.0
WORLD_DETERMINISTIC = False

# -------------------------
# Output Paths
# -------------------------
CONTROLLER_SAVE_DIR = "artifacts/controller"
CONTROLLER_BEST_PATH = f"{CONTROLLER_SAVE_DIR}/controller_best.pt"

# -------------------------
# Misc
# -------------------------
SEED = 42
