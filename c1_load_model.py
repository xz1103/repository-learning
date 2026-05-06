from transformers import AutoModelForCausalLM, AutoTokenizer

# =================================== LLM模型的加载=========================
# 加载模型和 tokenizer （第一次加载的时候会进行下载）
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
# print(model)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# tokenizer.special_tokens_map

# 构建pipline
from transformers import pipeline
# step1: 生成 pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,  # 只返回新生成的文本部分，不包含输入的prompt
    max_new_tokens=500,      # 模型生成的最大token
    do_sample=False
)

# step2: 构建 prompt
messages = [
    {"role": "user", "content": "写一个和猫有关的笑话."}
]

# step3 输出模型回复
output = generator(messages)
print(output[0]["generated_text"])