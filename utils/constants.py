from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent

# Now DATA_PATH is universally locked to four-pixel-vio/data
DATA_PATH = ROOT_PATH / "data"

MODEL_PATH = DATA_PATH / "models"
LOG_PATH = DATA_PATH / "logs"
EXP_CONFIGS_PATH = DATA_PATH / "configs"

DIODE_DIRECTIVITY_PATH = DATA_PATH / "S9119-01-directivity.mat"

# Create directories if they do not already exist (safely targets the true root)
MODEL_PATH.mkdir(parents=True, exist_ok=True)
LOG_PATH.mkdir(parents=True, exist_ok=True)
EXP_CONFIGS_PATH.mkdir(parents=True, exist_ok=True)