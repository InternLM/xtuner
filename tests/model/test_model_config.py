import tempfile
import json
from pathlib import Path
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig as HFQwen3MoeConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config

def test_hf_config_has_torch_dtype():
    # 为了兼容lmdeploy等推理引擎, hf_config导出必须有dtype字段
    cfg = Qwen3MoE30BA3Config()
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.save_hf(tmpdir)
        hf_path = Path(tmpdir)
        hf_dict = json.load(open(hf_path / "config.json", "r"))
        assert hf_dict["dtype"] == "bfloat16"


def test_qwen3_moe_auto_map(tmp_path):
    cfg = Qwen3MoE30BA3Config()
    cfg.auto_map = {
        "AutoConfig": "configuration_qwen_fope.Qwen3MoeConfig",
        "AutoModel": "modeling_qwen_fope.Qwen3MoeForCausalLM",
        "AutoModelForCausalLM": "modeling_qwen_fope.Qwen3MoeForCausalLM",
    }
    config_path = tmp_path/"config.json"
    cfg.save_hf(config_path)

    hf_config = HFQwen3MoeConfig.from_pretrained(config_path)
    assert hf_config.auto_map == cfg.auto_map
