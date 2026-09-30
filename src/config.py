import os


# ============================================================
# PROJECT PATHS
# ============================================================

DATASET_DIR = "fashion-dataset"

IMAGES_DIR = os.path.join(
    DATASET_DIR,
    "images"
)

STYLES_DIR = os.path.join(
    DATASET_DIR,
    "styles"
)

IMAGES_CSV = os.path.join(
    DATASET_DIR,
    "images.csv"
)

STYLES_CSV = os.path.join(
    DATASET_DIR,
    "styles.csv"
)


# ============================================================
# GENERATED DATA
# ============================================================

DATA_DIR = "data"

PRODUCTS_CSV = os.path.join(
    DATA_DIR,
    "products.csv"
)


# ============================================================
# EMBEDDINGS
# ============================================================

EMBEDDINGS_DIR = "embeddings"

PRODUCT_EMBEDDINGS = os.path.join(
    EMBEDDINGS_DIR,
    "product_embeddings.npy"
)

PRODUCT_IDS = os.path.join(
    EMBEDDINGS_DIR,
    "product_ids.npy"
)


# ============================================================
# FAISS
# ============================================================

INDEX_DIR = "index"

FAISS_INDEX = os.path.join(
    INDEX_DIR,
    "fashion.index"
)


# ============================================================
# CLIP
# ============================================================

MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cpu"


# ============================================================
# SEARCH
# ============================================================

RETRIEVAL_K = 50

FINAL_K = 10