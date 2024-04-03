准备对话模版
============

大模型的微调、对话均需要选择一个合适的对话模版（prompt
template）。XTuner
设计了一套对话模版封装逻辑，并提供了一系列社区广泛使用的对话模版。

本文将从“何处需要对话模版？”、“XTuner
内置对话模版速览”、“如何选择对话模版？”、“如何自定义对话模版？”四部分展开介绍。

何处需要对话模版？
------------------

   | 请确保在训练、对话和自定义应用场景中，始终保持对话模板的一致，否则可能会出现不符合预期的结果。

-  XTuner 内部

   -  训练（\ ``xtuner train``\ ）：需要使用对话模版将训练数据“模版化”

      -  在训练 ``config`` 中配置 ``prompt_template`` 参数来选择对话模版

   -  对话（\ ``xtuner chat``\ ）：需要使用对话模版将对话文本“模版化”

      -  利用 ``xtuner chat`` 命令的 ``--prompt-template``
         参数选择对话模版

-  其他自定义应用场景（如自定义 WebUI 等）

   -  需要按照训练所使用的对话模版，对对话文本进行“模版化”处理

XTuner 内置对话模版速览
-----------------------

XTuner 对现有大多数大语言模型的对话模版进行了实现，并集成在
``xtuner.utils.PROMPT_TEMPLATE`` 内，用户可以直接使用。

现已支持的对话模版有：\ ``internlm_chat``\ 、\ ``internlm2_chat``\ 、\ ``gemma``\ 、\ ``mistral``\ 、\ ``mixtral``\ 、\ ``qwen_chat``\ 、\ ``chatglm2``\ 、\ ``chatglm3``\ 、\ ``baichuan_chat``\ 、\ ``baichuan2_chat``\ 、\ ``deepseek_moe``\ 、\ ``deepseek_coder``\ 、\ ``vicuna``\ 、\ ``default``
等等。

代码结构
~~~~~~~~

以 ``internlm2_chat`` 模版为例，其代码结构如下。

.. code:: python

   internlm2_chat=dict(
       SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
       INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                    '<|im_start|>assistant\n'),
       SUFFIX='<|im_end|>',
       SUFFIX_AS_EOS=True,
       SEP='\n',
       STOP_WORDS=['<|im_end|>']),

-  ``SYSTEM``\ ：表示问答时“系统”字段的模版，其中 ``{system}``
   指代“系统”文本。值得注意的是，该字段在多轮对话中只会出现一次，即在第一轮。

-  ``INSTRUCTION``\ ：表示问答时“指令”字段的模版，其中 ``{input}``
   指代用户指令文本。

-  ``SUFFIX``\ ：表示“指令”字段的后缀，将会追加在每一轮问答的“回答”后面。通常，这也是一个特殊的结束符号。默认是空串\ ``''``\ 。

-  ``SUFFIX_AS_EOS``\ ：表示上述后缀是否作为结束符号。如果为
   ``True``\ ，则会取代 ``tokenizer`` 的 ``eos_token``\ ，否则，仍会使用
   ``tokenizer`` 的 ``eos_token`` 表示结束符号。默认是 ``False``\ 。

-  ``SEP``\ ：用于间隔多轮对话，将会追加在 ``INSTRUCTION`` 和 ``SUFFIX``
   后面。默认是空串\ ``''``\ 。

-  ``STOP_WORDS``\ ：用于指明结束词，该信息将被用在文本生成阶段。值得注意的是，\ ``tokenizer``
   的 ``eos_token`` 会被自动添加到 ``STOP_WORDS``\ ，而无需手动配置。

模版化结果
~~~~~~~~~~

以 ``internlm2_chat`` 模版为例，其对应的单轮、多轮模版化结果如下。

**单轮**

.. code::

   <|im_start|>system
   XXXXXXXXXXXXXXXXXXXXXXXX<|im_end|>
   <|im_start|>user
   YYYYYYYYYYYYYYYYYYYYYYYY<|im_end|>
   <|im_start|>assistant
   ZZZZZZZZZZZZZZZZZZZZZZZZ<|im_end|>

**多轮**

.. code::

   <|im_start|>system
   XXXXXXXXXXXXXXXXXXXXXXXX<|im_end|>
   <|im_start|>user
   YYYYYYYYYYYYYYYYYYYYYYYY<|im_end|>
   <|im_start|>assistant
   ZZZZZZZZZZZZZZZZZZZZZZZZ<|im_end|>
   <|im_start|>user
   YYYYYYYYYYYYYYYYYYYYYYYY<|im_end|>
   <|im_start|>assistant
   ZZZZZZZZZZZZZZZZZZZZZZZZ<|im_end|>

如何选择对话模版？
------------------

选择准确的对话模版是训练、应用模型的关键。关于如何选择对话模版，我们建议：

-  微调 chat 模型

   -  使用模型所对应的对话模版，如 ``internlm2-chat`` 使用
      ``internlm2_chat``\ 、\ ``Qwen-Chat`` 使用 ``qwen_chat``\ 。

-  微调 base 模型

   -  全量微调：任选对话模版，优先使用 chat
      版模型所对应的对话模版或默认对话模版 ``default``\ 。

      -  注：使用 chat 版模型所对应的对话模版请留意下方 “小贴士” 。

   -  LoRA / QLoRA 微调：使用默认对话模版 ``default``\ 。这是由于 LoRA /
      QLoRA 微调默认会关闭 ``embed_tokens`` 和 ``lm_head``
      的训练，此时如果引入未学习过的特殊 token（如对话模版中的
      ``<|im_start|>``\ ），则会影响模型的训练。

      -  注：通过修改 ``LoraConfig`` 可以引入 ``embed_tokens`` 和
         ``lm_head``
         的训练（会增大显存需求），进而支持任选对话模版（使用 chat
         版模型所对应的对话模版请留意下方 “小贴士” ）

         .. code:: diff

            lora=dict(
                type=LoraConfig,
                r=64,
                lora_alpha=16,
                lora_dropout=0.1,
                bias='none',
            +   modules_to_save=['embed_tokens', 'lm_head']  # 请确保与模型中所使用的参数名一致
                task_type='CAUSAL_LM')

**小贴士**

-  大多数的 base 模型所使用的 tokenizer 中不包含 chat
   模型对话模板中所使用的特殊 token 编码（例如 `internlm2
   chat <https://huggingface.co/internlm/internlm2-chat-1_8b/blob/ecccbb5c87079ad84e5788baa55dd6e21a9c614d/tokenizer_config.json#L29-L85>`__
   和 `internlm2
   base <https://huggingface.co/internlm/internlm2-1_8b/blob/main/tokenizer_config.json>`__\ ）。因此，如果要微调
   base 模型并配合使用 chat 版对话模版，需确保在 Config
   中及后续全流程使用 chat 版模型的 tokenizer。Config 中修改 tokenizer
   的方式为：

   .. code:: diff

      tokenizer = dict(
          type=AutoTokenizer.from_pretrained,
      -   pretrained_model_name_or_path=pretrained_model_name_or_path,
      +   pretrained_model_name_or_path='PATH_TO_CHAT_LLM_TOKENIZER',
          trust_remote_code=True,
          padding_side='right')

如何自定义对话模版？
--------------------

如果 XTuner
所内置的对话模版不能满足实际需求，用户可以实现自定义的对话模版。

具体来说，可以在
`template.py <https://github.com/InternLM/xtuner/blob/main/xtuner/utils/templates.py>`__
的 ``PROMPT_TEMPLATE`` 中新增一个对话模版，并参考 “XTuner
内置对话模版速览” 章节对每个字段的描述进行自定义修改。

附：XTuner 内置 configs 所选择的对话模版
----------------------------------------

======================================== ==============
模型                                     对话模版
======================================== ==============
baichuan-inc/Baichuan-7B                 default\*
baichuan-inc/Baichuan-13B-Base           default\*
baichuan-inc/Baichuan-13B-Chat           baichuan_chat
baichuan-inc/Baichuan2-7B-Base           default\*
baichuan-inc/Baichuan2-7B-Chat           baichuan2_chat
baichuan-inc/Baichuan2-13B-Base          default\*
baichuan-inc/Baichuan2-13B-Chat          baichuan2_chat
THUDM/chatglm2-6b                        chatglm2
THUDM/chatglm3-6b                        chatglm3
THUDM/chatglm3-6b-base                   chatglm3
deepseek-ai/deepseek-coder-6.7b-base     deepseek_coder
deepseek-ai/deepseek-coder-6.7b-instruct deepseek_coder
internlm/internlm-7b                     default\*
internlm/internlm-20b                    default\*
internlm/internlm-chat-7b                internlm_chat
internlm/internlm-chat-20b               internlm_chat
huggyllama/llama-7b                      default
meta-llama/Llama-2-7b-hf                 llama2_chat
meta-llama/Llama-2-7b-chat-hf            llama2_chat
meta-llama/Llama-2-70b-hf                llama2_chat
lmsys/vicuna-7b-v1.5                     vicuna
lmsys/vicuna-13b-v1.5                    vicuna
mistralai/Mistral-7B-v0.1                mistral
mistralai/Mixtral-8x7B-v0.1              mixtral
mistralai/Mixtral-8x7B-Instruct-v0.1     mixtral
Qwen/Qwen-1_8B                           default\*
Qwen/Qwen-1_8B-Chat                      qwen_chat
Qwen/Qwen-7B                             default\*
Qwen/Qwen-7B-Chat                        qwen_chat
Qwen/Qwen-72B                            default\*
Qwen/Qwen-72B-Chat                       qwen_chat
bigcode/starcoder                        default
01-ai/Yi-6B                              default
01-ai/Yi-34B                             default
HuggingFaceH4/zephyr-7b-beta             zephyr
deepseek-ai/deepseek-moe-16b-base        deepseek_moe
deepseek-ai/deepseek-moe-16b-chat        deepseek_moe
internlm/internlm2-1_8b                  default\*
internlm/internlm2-7b                    default\*
internlm/internlm2-20b                   default\*
internlm/internlm2-chat-1_8b             internlm2_chat
internlm/internlm2-chat-7b               internlm2_chat
internlm/internlm2-chat-20b              internlm2_chat
Qwen/Qwen1.5-0.5B                        default\*
Qwen/Qwen1.5-0.5B-Chat                   qwen_chat
Qwen/Qwen1.5-1.8B                        default\*
Qwen/Qwen1.5-1.8B-Chat                   qwen_chat
Qwen/Qwen1.5-4B                          default\*
Qwen/Qwen1.5-4B-Chat                     qwen_chat
Qwen/Qwen1.5-7B                          default\*
Qwen/Qwen1.5-7B-Chat                     qwen_chat
Qwen/Qwen1.5-14B                         default\*
Qwen/Qwen1.5-14B-Chat                    qwen_chat
Qwen/Qwen1.5-72B                         default\*
Qwen/Qwen1.5-72B-Chat                    qwen_chat
google/gemma-2b                          default\*
google/gemma-2b-it                       gemma
google/gemma-7b                          default\*
google/gemma-7b-it                       gemma
======================================== ==============

\*: 官方对话模版中存在特殊 token（比如
``<|im_start|>``\ 、\ ``<|im_end|>``\ ），这类特殊 token
在预训练阶段并未得到训练。故，使用 ``default`` 模版。
