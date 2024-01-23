######################################################################
#              생성형 언어 모델 Instruction Tunning 해보기
#
# [install pre-requisites]
# 기본적으로 아래 패키지 설치
# pip install bitsandbytes==0.41.1
# pip install accelerate==0.21.0
# pip install appdirs
# pip install loralib
# pip install datasets
# pip install fire
# pip install git+https://github.com/huggingface/peft
# pip install transformers==4.34.1
# pip install sentencepiece sentence_transformers
# pip install scipy numpy scikit-learn pandas
# 
# * 버전 충돌 생기면 아래 모듈부터 upgrade
# pip install --upgrade huggingface_hub
#
# [download pre-trained model]
# git clone git@hf.co:upstage/SOLAR-10.7B-v1.0
# 
# [download training data]
# git clone git@hf.co:datasets/kyujinpy/Open-platypus-Commercial
#
######################################################################
import os
import os.path as osp
import sys
import fire
import json
from typing import List, Union

import torch
from torch.nn import functional as F

import transformers
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict
)
from peft import PeftModel

######################################################################
# (1) 필요한 모듈들 잘 설치되었는지 확인해보기
######################################################################
print('[Info] Required python modules are loaded !!!')


######################################################################
# (2) Pre-trained 모델 가지고와서 fine tuning 위한 설정하기
# - SOLAR-10.7B 는 Llama2 구조를 확장한 것임 AutoRegressive 방식
# - 모델 가중치는 8bit 로 초기 로드
# - 계산 과정에서는 float16 형식을 사용 
######################################################################
device = 'auto' 
# base_LLM_model = 'upstage/SOLAR-10.7B-v1.0' 
base_LLM_model = '/content/drive/MyDrive/llm/SOLAR-10.7B-v1.0'
model = AutoModelForCausalLM.from_pretrained(   # SOLAR-10.7B 는 CasualModeling, AR 방식
    base_LLM_model,                             # 모델 위치 (hugging face 로그인 사전에 해두어야 함)
    load_in_8bit=True,                          # LoRA 를 위해서 8 비트로 로딩
    torch_dtype=torch.float16,                  # 연산에 사용할 데이터는 float16 
    device_map=device)                          # GPU, CPU 등은 알아서 선택
tokenizer = AutoTokenizer.from_pretrained(base_LLM_model)
print('[Info] Model is downloaded & loaded !!!')

######################################################################
# (3) 입력 데이터를 위한 토큰, 모델 구성 정보 확인
######################################################################
# Check special token
bos = tokenizer.bos_token_id # 문장 시작 토큰
eos = tokenizer.eos_token_id # 문장 끝 토큰
pad = tokenizer.pad_token_id # 문장 패딩 토큰
tokenizer.padding_side = "right" # 패딩 오른쪽

print("[Info] BOS token:", bos) 
print("[Info] EOS token:", eos) 
print("[Info] PAD token:", pad) 

if (pad == None) or (pad == eos):
    tokenizer.pad_token_id = 0  
print("[Info] length of tokenizer:",len(tokenizer)) 

print(f'[Info] Model Information \n {model}')
print(f'[Info] Model Type Information \n {type(model)}')


######################################################################
# (4) Instruct Tuning 할 데이터 확보하기
######################################################################
# Instruction Tuning 위한 프롬프트 구조 및 템플릿 정의하기
# - 프롬프트의 구조를 설명하는 description 을 주고 instruction, input, reponse 로 구조를 정의
instruct_template = {
    "prompt_input": '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n
                    ### Instruction:\n{instruction}\n\n
                    ### Input:\n{input}\n\n
                    ### Response:\n''',
                    
    "prompt_no_input": '''Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n
                    ### Instruction:\n{instruction}\n\n
                    ### Response:\n''',
                    
    "response_split": "### Response:"
}


# Prompt 템플릿을 받아서 여기에 isntruction, input 을 채우는 util 
class Prompter(object):
    def __init__(self, verbose: bool = False):
        self.template = instruct_template

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:

        if input: # input text가 있다면
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )

        if label:
            res = f"{res}{label}"

        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

prompter = Prompter()


######################################################################
# (5) 입력 prompt 를 모델 입력 token 으로 만들기
######################################################################
# Tokenizer 설정
cutoff_len = 4096
train_on_inputs = False
add_eos_token = False

# Tokenizing
# - Prompt 를 가지고 모델에 입력하기 위해 처리
def tokenize(prompt, add_eos_token=True):
    # tokernizer 설정해서 가지고 오기 (cutoff_len 은 하이퍼파라미터 이용)
    result = tokenizer( prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None,)

    # EOS 토큰 삽입해두고 Note 하기
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id   # EOS 토큰 없고
        and len(result["input_ids"]) < cutoff_len           # 길이가 cutoff_len 보다 작고
        and add_eos_token                                   # EOS 붙이기로 했으면
    ):
        # EOS 토큰을 붙이고 이 EOS 토큰에 집중(attention)하라고 1 로 설정
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    # 모델이 맞춰야 하는것이 입력임 (Auto Regressive 방식의 특징)
    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    # tokenizing 하기
    full_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"], data_point["output"])
    tokenized_full_prompt = tokenize(full_prompt)
    
    # input 이 없는 경우
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"])
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1
        
        # -100 으로 채워두어 학습할 때 무시하라고 마킹
        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        
    return tokenized_full_prompt


######################################################################
# (6) 학습할 데이터 로드하고 Tokenizing 해서 모델 입력 준비해놓기
######################################################################
# data = load_dataset('kyujinpy/Open-platypus-Commercial')
data = load_dataset('/content/drive/MyDrive/llm/Open-platypus-Commercial')
print(f'[Info] Train Data Length {len(data["train"])}')
print(f'[Info] Train Data looks like :\n{data["train"][1:10]}')
train_data = data["train"].shuffle() 
train_data = train_data.map(generate_and_tokenize_prompt)


######################################################################
# (7) LoRA 방식 설정 및 적용
######################################################################
# LoRA 설정
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
lora_target_modules = ["gate_proj", "down_proj", "up_proj"]
config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM")

# model 에 LoRA 적용
model = prepare_model_for_int8_training(model)
model = get_peft_model(model, config) 


######################################################################
# (8) 학습기 설정
######################################################################
# 주요 하이퍼파라미터
batch_size = 16
num_epochs = 1
micro_batch = 1 # GPU 1개만 쓸거라서 나누지 않음
gradient_accumulation_steps = batch_size // micro_batch
lr_scheduler = 'cosine'
warmup_ratio = 0.06 
learning_rate = 4e-4
optimizer = 'adamw_torch'
weight_decay = 0.01
max_grad_norm = 1.0

# Trainer 만들기
output_dir='/content/drive/MyDrive/llm/customized'
trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=None,
        args=transformers.TrainingArguments( 
            per_device_train_batch_size = micro_batch,
            gradient_accumulation_steps = gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim=optimizer,
            evaluation_strategy="no",
            save_strategy="steps",
            max_grad_norm = max_grad_norm,
            save_steps = 30, 
            lr_scheduler_type=lr_scheduler,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            group_by_length = False
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

######################################################################
# (8) 학습하기
######################################################################
resume_from_checkpoint = '/content/drive/MyDrive/llm/customized/checkpoint-150'
if resume_from_checkpoint:
    checkpoint_name = os.path.join(
        resume_from_checkpoint, "pytorch_model.bin"
    )  # All checkpoint

    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "adapter_model.bin"
        )  
        resume_from_checkpoint = (
            True
        ) 

    if os.path.exists(checkpoint_name):
        print(f"[Info] Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"[Error] Checkpoint {checkpoint_name} not found")

model.config.use_cache = False

# 학습규모 파악해보기
print('[Info] Parameter volume to be trained')
model.print_trainable_parameters() 

# 학습하기
model = torch.compile(model)
torch.cuda.empty_cache()
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# 저장하기
model.save_pretrained(output_dir)
model_path = os.path.join(output_dir, "pytorch_model.bin")
torch.save({}, model_path)
tokenizer.save_pretrained(output_dir)