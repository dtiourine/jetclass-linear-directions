from pathlib import Path

# Model
INPUT_DIM = 17
MODEL_DIM = 512
NUM_LAYERS = 6
NUM_HEADS = 8
NUM_CLASSES = 10
MAX_PARTICLES = 128

# Training
SEED = 42
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
DROPOUT = 0.1
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
MAX_GRAD_NORM = 1.0
BUFFER_SIZE = 20  # ROOT files buffered & shuffled together for class mixing 
CHECKPOINT_EVERY_STEPS = 5000
LOG_EVERY_STEPS = 50           
EVAL_EVERY_STEPS = 10000

# OneCycle LR schedule
WARMUP_PCT = 0.1           
LR_DIV_FACTOR = 25.0       
LR_FINAL_DIV_FACTOR = 1e4

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]

ROOT_DATA_DIR = PROJ_ROOT / "data"
TRAIN_DATA_DIR = PROJ_ROOT / "data" / "train"
VAL_DATA_DIR = PROJ_ROOT / "data" / "val"
TEST_DATA_DIR = PROJ_ROOT / "data" / "test"

CHECKPOINT_DIR = PROJ_ROOT / "checkpoints"
LOG_DIR = PROJ_ROOT / "runs"  
