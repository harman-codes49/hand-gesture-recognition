import os

# Paths
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data", "leapGestRecog")
MODEL_PATH    = os.path.join(BASE_DIR, "models", "gesture_model.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "models", "label_map.json")
HISTORY_PATH  = os.path.join(BASE_DIR, "models", "training_history.json")

# Image settings
IMG_SIZE    = 96
IMG_SHAPE   = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE  = 32

# Training
EPOCHS_FROZEN   = 20   # phase 1 – only top layers
EPOCHS_FINETUNE = 30   # phase 2 – unfreeze top 30 base layers
LEARNING_RATE   = 1e-3
FINETUNE_LR     = 1e-4
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.10
UNFREEZE_LAYERS = 30   # how many top base-model layers to unfreeze

# Gesture metadata for LeapGestRecog
# Keys are string folder-prefix numbers so they sort naturally
GESTURE_FOLDER_MAP = {
    "01_palm":       "Palm",
    "02_l":          "L",
    "03_fist":       "Fist",
    "04_fist_moved": "Fist Moved",
    "05_thumb":      "Thumb",
    "06_index":      "Index",
    "07_ok":         "OK",
    "08_palm_moved": "Palm Moved",
    "09_c":          "C",
    "10_down":       "Down",
}

GESTURE_EMOJIS = {
    "C":         "🤏",
    "Index":     "☝️",
    "L":         "🤙",
    "Three":     "🖖",
    "Four":      "🖐️",
    "Palm":      "✋",
    "OK":        "👌",
    "Rock":      "🤘",
    "ILoveYou":  "🤟",
    "Spock":     "🖖",
    "Down":      "👇",
    "PalmSide":  "🤚",
    "Point":     "👉",
    "One":       "☝️",
    "Fist":      "✊",
    "Grip":      "✊",
    "Thumb":     "👍",
    "Hang":      "🤙",
    "PointDown": "👇",
    "ThumbUp":   "👍",
}

NUM_CLASSES = len(GESTURE_FOLDER_MAP)
