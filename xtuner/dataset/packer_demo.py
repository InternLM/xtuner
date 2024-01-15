from transformers import AutoTokenizer
from utils import Packer, encode_fn

# 使用 AutoTokenizer 创建一个 Tokenizer 实例
tokenizer = AutoTokenizer.from_pretrained("/public/home/lvshuhang/model_space/workspace/THUDM_chatglm3-6b-base/", trust_remote_code=True)

# 假设的输入数据
example = {
    'conversation': [
        {
            'input': '今天的天气怎么样？',
            'output': '今天的天气非常不错，是晴天。'
        },
        {
            'input': '你今天吃饭了吗？',
            'output': '我今天吃饭了，吃的蔬菜沙拉。'
        },
        {
            'input':'这是最后一句话，需要足够长以达到最大长度限制，以触发特殊处理逻辑。',
            'output':'确实，这句话确实足够长了。'
        }

    ],
        'conversation': [
        {
            'input': 'Give three tips for staying healthy.',
            'output': '1.Eat a balanced diet xxx'
        }


    ]
}


# 假设的最大长度
max_length = 10  # 这个值可以根据实际情况调整

# 编码数据
encoded_example = encode_fn(example, tokenizer, max_length, input_ids_with_output=True, with_image_token=False)

# 打印编码后的结果
print("Encoded example:", encoded_example,len(encoded_example['input_ids']))

# 使用 Packer 处理编码后的数据
packer = Packer(chunk_size=max_length)
packed_data = packer([encoded_example])

# 打印处理后的结果
print("Packed data:", packed_data)