import random
import numpy as np
import torch


def set_all_seeds(seed=42):
    """
    Set all seeds for reproducibility (PyTorch, NumPy, Python's random, etc.)

    Args:
        seed (int): The seed value to use
    """
    # Python's random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set seeds for sklearn if used
    try:
        import sklearn

        sklearn.utils.check_random_state(seed)
    except:
        pass

    print(f"All seeds have been set to {seed}")
