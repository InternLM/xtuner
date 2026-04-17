from copy import deepcopy
from typing import Callable

from lagent.agents import AsyncAgent

from xtuner.v1.data_proto.rl_data import (
    RLDataFlowItem,
    RLJudgerResponseItem,
    RLRolloutResponseItem,
    RolloutState,
    update_dataflow_item,
)
from xtuner.v1.ray.environment.lagent.schema import AgentMessage
from xtuner.v1.ray.judger.controller import JudgerController


class JudgerWrapper(AsyncAgent):
    def __init__(
        self,
        judger_cfg=None,
        placement_group=None,
        judger_controller=None,
        itemgetter: Callable[[AgentMessage], RLDataFlowItem] = lambda m: m.content,
        reward_key: str = 'score',
        name: str = None,
    ):
        assert judger_controller is not None or (
            judger_cfg and placement_group
        ), "Either judger_controller or judger_cfg and placement_group must be provided."
        self.judger_controller = judger_controller or JudgerController.remote(judger_cfg, placement_group)
        self.itemgetter = itemgetter
        self.reward_key = reward_key
        super().__init__(memory=None, aggregator=None, name=name)

    async def forward(self, message: AgentMessage, meta_message: AgentMessage, *args, **kwargs) -> AgentMessage:
        item = deepcopy(self.itemgetter(meta_message))
        if isinstance(item, dict):
            item = RLDataFlowItem.model_validate(item)
        item = update_dataflow_item(
            [item],
            'env.rollout',
            [
                RLRolloutResponseItem(
                    response=message.content,
                    response_ids=message.content_ids,
                    logprobs=message.content_logprobs,
                    finish_reason='finished',
                    state=RolloutState.COMPLETED,
                )
            ],
        )[0]
        judger_response: RLJudgerResponseItem = await self.judger_controller.run.remote(item)
        return AgentMessage(sender=self.name, content=None, reward=judger_response.reward[self.reward_key])
