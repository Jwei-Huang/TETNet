import os, random
import numpy as np
import tensorflow as tf

def set_global_determinism(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # older TF builds may not support; ignore
        pass
