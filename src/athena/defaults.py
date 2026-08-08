from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_ARTIFICIAL_PKL = RAW_DATA_DIR / "Tell2Design_artificial_all.pkl"
DEFAULT_IMAGES_DIR = RAW_DATA_DIR / "floorplan_image"
DEFAULT_CLEANED_CSV = PROCESSED_DATA_DIR / "cleaned_annotations.csv"
DEFAULT_TOKENIZED_TEXTS = PROCESSED_DATA_DIR / "tokenized_texts.pt"
DEFAULT_PROCESSED_IMAGES = PROCESSED_DATA_DIR / "processed_images.pt"
DEFAULT_CHECKPOINT = MODELS_DIR / "athena_trained.pth"
DEFAULT_LOSS_LOG = LOGS_DIR / "loss_log.json"
DEFAULT_GENERATED_IMAGE = OUTPUTS_DIR / "generated_blueprint.png"


def resolve_device(requested=None):
    import torch

    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
