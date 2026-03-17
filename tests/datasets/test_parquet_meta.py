"""Tests for parquet save/load roundtrip (save_mixed_dict_to_parquet / load_mixed_dict_from_parquet)."""
import numpy as np


def test_parquet_meta_roundtrip(tmp_path):
    """jsonl_meta 从 npz 切到 parquet 后，保证 save/load 一致性."""
    from xtuner.v1.datasets.jsonl import save_mixed_dict_to_parquet, load_mixed_dict_from_parquet

    data = {
        # 1D 与 2D numpy 数组
        "a": np.array([1, 2, 3], dtype=np.int64),
        "b": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        # 规则嵌套 list（会在 roundtrip 中变成 ndarray 也可以）
        "c": [[1, 2], [3, 4, 5]],
        "d": [[1.5, 2.5], [3.5, 4.5]],
    }

    path = tmp_path / "jsonl_meta.parquet"
    save_mixed_dict_to_parquet(data, str(path))
    loaded = load_mixed_dict_from_parquet(str(path))

    # 逐字段对比，关注数值与形状，而不过分约束具体 dtype / 容器类型
    for k, v in data.items():
        orig_arr = v
        loaded_arr = loaded[k]
        if isinstance(orig_arr, np.ndarray):
            assert isinstance(loaded_arr, np.ndarray)
            assert orig_arr.shape == loaded_arr.shape
            assert np.allclose(orig_arr, loaded_arr)
        elif isinstance(orig_arr, list):
            assert isinstance(loaded_arr, list)
            assert orig_arr == loaded_arr
