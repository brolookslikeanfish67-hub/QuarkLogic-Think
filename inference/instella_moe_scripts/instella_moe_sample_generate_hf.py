###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "amd/Instella-MoE-16B-A3B-Think"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", trust_remote_code=True)

prompt = [{"role": "user", "content": "What are the computational benefits of Mixture-of-Experts models?"}]
inputs = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    return_tensors='pt'
)

tokens = model.generate(
    inputs.to(model.device),
    max_new_tokens=1024,
    temperature=0.6,
    top_p=0.95,
    do_sample=True
)

print(tokenizer.decode(tokens[0], skip_special_tokens=False))