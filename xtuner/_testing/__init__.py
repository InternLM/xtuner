from .glm52_hf import apply_glm52_hf_numeric_oracle_patch, load_glm52_hf_oracle_model
from .patch_hf import patch_hf_rms_norm, patch_hf_rope
from .utils import enable_full_determinism
from .testcase import DeterministicDDPTestCase
