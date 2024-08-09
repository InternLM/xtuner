## pipeline优化
### 优化原理

RLHF的每次迭代过程可以分为三个阶段：Generation、Forward和Train。在Generation阶段，由vLLM推理生成回复；在Forward阶段，actor、critic、reference和reward四个模型进行推理；在Train阶段，actor和critic模型进行训练。

在每个阶段运行时，其它阶段的GPU会处于空闲等待状态，导致资源浪费。

为了解决这个问题，可以借助流水线并行的思想进行优化。将batch数据分为多个小的micro-batch，每个阶段处理完一个micro-batch后，立即将数据传递到下一个阶段进行处理，而不是等待整个batch处理完成。这样可以减少各阶段GPU的空闲等待时间，提高资源利用率。

### 运行步骤

1）vLLM添加接口
- 获取vLLM安装路径
    ```shell
    export vllm=$(pip show numpy | grep Location | awk '{print $2"/vllm"}')
    ```

- 编辑$vllm/entrypoints/llm.py，在`class LLM`中添加下面两个接口
    ```python
    def generate_to_queue(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        prefix_pos: Optional[Union[int, List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        queue = None,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts and put result to queue.
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                                "provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                                "must be the same.")
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        # Add requests to the engine.
        num_requests = len(prompts) if prompts is not None else len(
            prompt_token_ids)
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            prefix_pos_i = prefix_pos[i] if prefix_pos is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[
                i]
            self._add_request(prompt,
                                sampling_params,
                                token_ids,
                                lora_request=lora_request,
                                prefix_pos=prefix_pos_i)
        return self._run_engine_to_queue(use_tqdm, queue)


    def _run_engine_to_queue(self, use_tqdm: bool, queue) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")
        # Run the engine.
        outputs: List[RequestOutput] = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    queue.put(output)
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs
    ```

### 参数配置
参考配置文件 examples/rlhf/internlm2_20b_pipe_32gpu.py
```python
...
PIPE_MICRO_BATCH_NUM = 4    # 调整micro-batch的数量
...
```

### 精度影响
启用norm_rewards时，精度无法严格对齐。原因在于norm_rewards对奖励进行了归一化处理。在优化前，归一化操作是在整个batch上进行的；而优化后，归一化操作是在每个micro-batch上分别进行。
