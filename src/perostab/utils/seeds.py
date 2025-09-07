import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        env_seed = os.getenv("PEROSTAB_SEED")
        seed = int(env_seed) if env_seed is not None else 42
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass
    return seed

