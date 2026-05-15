# src/config.example.py
# Copy this file to config.py and update paths for your machine.
# config.py is gitignored — never push it.

# --- WINDOWS EXAMPLE ---
# USB = "D:/Rabia_Uni/Semester_4/Programming_for_AI/Projects/Project_2/Dataset"

# --- LINUX EXAMPLE ---
# USB = "/media/your-username/YOUR-USB-NAME/Dataset"

USB = "YOUR_DATASET_ROOT_PATH_HERE"

STATIC_TRAIN   = f"{USB}/1_UAlpha40_Mendeley/static_signs/train"
STATIC_TEST    = f"{USB}/1_UAlpha40_Mendeley/static_signs/test"
DYNAMIC_RAW    = f"{USB}/1_UAlpha40_Mendeley/dynamic_signs/raw_videos"
WORD_TRAIN     = f"{USB}/2_PSL_Dictionary_Toolkit/train"
WORD_TEST      = f"{USB}/2_PSL_Dictionary_Toolkit/test"

FLOW_OUTPUT    = "YOUR_LOCAL_PROJECT_PATH/data/optical_flow"
MODEL_SAVE_DIR = "YOUR_LOCAL_PROJECT_PATH/models/"

IMG_SIZE       = (64, 64)
N_FRAMES       = 16
BATCH_SIZE     = 16
EPOCHS         = 30
