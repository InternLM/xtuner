from xtuner.v1._writer.jsonl_writer import JsonlWriter
import threading
import json


class TestJsonlWriter:
    def test_jsonl_writer(self, tmp_path):
        file_path = tmp_path / "test.jsonl"
        writer = JsonlWriter(file_path)

        for i in range(10):
            writer.add_scalar(tag="loss", scalar_value=i * 0.1, global_step=i)

        writer.close()
        with open(writer.log_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 10
            for i, line in enumerate(lines):

                data = json.loads(line)
                assert data == {"loss": i * 0.1, "step": i}

    def test_multi_threaded_jsonl_writer(self, tmp_path):
        file_path = tmp_path / "test_multi_threaded.jsonl"
        writer = JsonlWriter(file_path)

        def write_data(start, end):
            for i in range(start, end):
                writer.add_scalar(tag="accuracy", scalar_value=i * 0.01, global_step=i)

        threads = []
        for i in range(5):
            t = threading.Thread(target=write_data, args=(i * 20, (i + 1) * 20))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        writer.close()
        with open(writer.log_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 100
            steps = set()
            for line in lines:
                data = json.loads(line)
                steps.add(data["step"])
                assert data["accuracy"] == data["step"] * 0.01
            assert steps == set(range(100))
