import unittest

import numpy as np
from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore, TokenizedSegment, Trie


class ProvenanceTests(unittest.TestCase):
    def test_three_versions_append_without_replacing_old_routes_or_probabilities(self):
        store = RolloutTraceStore.__ray_metadata__.modified_class()
        text = ""
        history = []
        for version in range(3):
            delta = f"prompt{version}"
            response = f"response{version}"
            routes = np.full((2, 2, 2), version)
            text += delta
            store.insert(
                "s",
                text,
                TokenizedSegment(
                    text=delta,
                    token_ids=[version * 10, version * 10 + 1],
                    expert_key=routes.copy(),
                    request_id=f"r{version}",
                    model_step=version,
                ),
            )
            text += response
            store.insert(
                "s",
                text,
                TokenizedSegment(
                    text=response,
                    token_ids=[version * 10 + 2, version * 10 + 3],
                    labels=[version * 10 + 2, version * 10 + 3],
                    logprobs=[-0.1 - version, -0.2 - version],
                    expert_key=routes.copy(),
                    request_id=f"r{version}",
                    model_step=version,
                ),
            )
            trace = store.export_training_trace("s", text)
            for old, new in zip(history, trace["routed_experts"]):
                np.testing.assert_array_equal(old, new)
            history = [a.copy() for a in trace["routed_experts"]]
        self.assertEqual(trace["response_model_steps"], [0, 0, 1, 1, 2, 2])
        self.assertEqual(trace["request_ids"], ["r0", "r1", "r2"])
        self.assertTrue(store.contains_request_ids("s", ["r0", "r1", "r2"]))
        self.assertFalse(store.contains_request_ids("s", ["forged"]))
        self.assertEqual(trace["logprobs"][:4], [0, 0, -0.1, -0.2])
        self.assertEqual(trace["input_ids"][:4], [0, 1, 2, 3])
        # Ordinary prefix reading remains legal; conflicting writes fail.
        self.assertTrue(store.search("s", text)[0])
        with self.assertRaisesRegex(ValueError, "old_request_id=r2.*new_request_id=other"):
            store.insert("s", text, TokenizedSegment(text="other", token_ids=[99], request_id="other", model_step=3))
        self.assertEqual(store.export_training_trace("s", text)["input_ids"], trace["input_ids"])

    def test_long_lived_trace_can_enumerate_and_release_without_recursion(self):
        trie = Trie()
        for i in range(3000):
            trie.insert("x" * (i + 1), i)
        self.assertEqual(len(trie.keys()), 3000)
        trie.release()
        trie.release()
        self.assertEqual(trie.keys(), [])
