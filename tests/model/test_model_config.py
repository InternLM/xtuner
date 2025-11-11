import tempfile
import json
from pathlib import Path
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config

def test_hf_config_has_torch_dtype():
    # 为了兼容lmdeploy等推理引擎, hf_config导出必须有dtype字段
    cfg = Qwen3MoE30BA3Config()
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.save_hf(tmpdir)
        hf_path = Path(tmpdir)
        hf_dict = json.load(open(hf_path / "config.json", "r"))
        assert hf_dict["dtype"] == "bfloat16"
