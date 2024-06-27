# 小论文-LLMopt区域供热系统LLM辅助调控优化
# -- coding: utf-8 --
# -------------------------------
# @Author : Ning Zhang
# @Email : zhang_n@zju.edu.cn
# -------------------------------
# @File : llm_lib.py
# @Time : 2024/6/26 下午12:01
# -------------------------------
# 调llm用模型

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed

def llm_call(model_path, prompt_input,max_length=1500, temperature=0.1):
    set_seed(42)
    print(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
    model = model.eval()

    inputs = tokenizer([prompt_input], return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    gen_kwargs = {"max_length": max_length,"repetition_penalty": 1.0}
    # gen_kwargs = {"max_length": max_length, "top_p": 0.8, "temperature": temperature, "do_sample": True,
                  # "repetition_penalty": 1.0}
    output = model.generate(**inputs, **gen_kwargs)
    output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print(output)

    return output
    

