try:
    from torch.utils.data import Dataset  # type: ignore
except Exception:
    class Dataset:  # fallback for environments without torch
        pass

