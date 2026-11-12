"""CPU boundary regression: real trace export, no Ray runtime or backend."""
import unittest
from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore, TokenizedSegment

class TracePromptBoundaryTest(unittest.TestCase):
    def store(self):
        cls=RolloutTraceStore.__ray_metadata__.modified_class
        return cls()

    def test_tools_and_versions_share_the_trace_boundary(self):
        s=self.store()
        s.insert("s","prompt",TokenizedSegment(text="fixture",token_ids=[1,2,3],labels=[-100]*3,logprobs=[0.]*3,model_step=0))
        s.insert("s","promptanswer",TokenizedSegment(text="fixture",token_ids=[4,5],labels=[4,5],logprobs=[-.1]*2,model_step=0,request_id="r0"))
        s.insert("s","promptanswertool",TokenizedSegment(text="fixture",token_ids=[6],labels=[-100],logprobs=[0.],model_step=1))
        s.insert("s","promptanswertoolanswer2",TokenizedSegment(text="fixture",token_ids=[7],labels=[7],logprobs=[-.2],model_step=1,request_id="r1"))
        data=s.export_training_trace("s","promptanswertoolanswer2")
        self.assertEqual(data["prompt_ids"],[1,2,3])
        self.assertEqual(data["response_ids"],[4,5,6,7])
        self.assertEqual(data["input_ids"],data["prompt_ids"]+data["response_ids"])
        self.assertEqual(data["token_model_steps"][len(data["prompt_ids"]):],[0,0,1,1])
        self.assertEqual(data["request_ids"],["r0","r1"])

    def test_reject_nonprompt_head(self):
        s=self.store()
        s.insert("s","bad",TokenizedSegment(text="fixture",token_ids=[1,2],labels=[-100,2],logprobs=[0.,-.1]))
        with self.assertRaisesRegex(ValueError,"not a prompt"):
            s.export_training_trace("s","bad")


class SandboxPromptBoundaryTest(unittest.IsolatedAsyncioTestCase):
    async def test_dataset_prompt_does_not_shift_versions(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch
        from xtuner.v1.data_proto.rl_data import RolloutState
        from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import AgentInSandboxLoop
        from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import AgentRolloutItem, RolloutStatus
        messages=[{"role":"user","content":"task"},{"role":"assistant","content":"answer"}]
        item=AgentRolloutItem(id="fixture",data_source="test",instruction="task.md",
            status=RolloutStatus.COMPLETED,reward=1.,
            artifacts={"messages":[{"messages":messages,"tools":[],"request_ids":["r0","r1"]}],
                       "response_message":{"content":"answer","finish_reason":"stop"}})
        state=RolloutState(message=[],prompt_ids=[99],session_id=1)
        data={"input_ids":[1,2,3,4,5,6,7],"labels":[-100,-100,-100,4,5,-100,7],
              "prompt_ids":[1,2,3],"response_ids":[4,5,6,7],"logprobs":[0.]*7,
              "routed_experts":["route0","route1"],"token_model_steps":[0,0,0,0,0,1,1],
              "request_ids":["r0","r1"]}
        store=SimpleNamespace(export_training_trace=SimpleNamespace(remote=AsyncMock(return_value=data)),
                              contains_request_ids=SimpleNamespace(remote=AsyncMock(return_value=True)))
        loop=AgentInSandboxLoop.__new__(AgentInSandboxLoop)
        loop.mode="train";loop.require_turn_provenance=True
        loop.tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:"prompt",decode=lambda x:"answer")
        with patch("xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop.get_store",return_value=store):
            result=await loop._build_rollout_states(state,item)
        self.assertEqual(len(result),1)
        sample=result[0]
        self.assertEqual(sample.prompt_ids,[1,2,3])
        self.assertEqual(sample.response_ids,[4,5,6,7])
        self.assertEqual(sample.response_model_steps,[0,0,1,1])
        self.assertEqual(sample.input_ids,sample.prompt_ids+sample.response_ids)
