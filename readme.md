# Practice - finetune generative AI models

References

* [How to Fine-Tune an LLM Part 1: Preparing a Dataset for Instruction Tuning | alpaca\_ft – Weights & Biases (wandb.ai)](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2#%F0%9F%91%89-continue-to-part-2:-training-our-llm)
* [How to Fine-Tune an LLM Part 2: Instruction Tuning Llama 2 | alpaca\_ft – Weights & Biases (wandb.ai)](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-2-Instruction-Tuning-Llama-2--Vmlldzo1NjY0MjE1)
* [How to Fine-tune an LLM Part 3: The HuggingFace Trainer | alpaca\_ft – Weights & Biases (wandb.ai)](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy)
* [Home – Weights & Biases (wandb.ai)](https://wandb.ai/home)
* [tcapelle/llm\_recipes: A set of scripts and notebooks on LLM finetunning and dataset creation (github.com)](https://github.com/tcapelle/llm_recipes)
* [Hugging Face – The AI community building the future.](https://huggingface.co/kyujinpy)
* [[2308.10792] Instruction Tuning for Large Language Models: A Survey (arxiv.org)](https://arxiv.org/abs/2308.10792)
* [xiaoya-li/Instruction-Tuning-Survey: Project for the paper entitled \`Instruction Tuning for Large Language Models: A Survey\` (github.com)](https://github.com/xiaoya-li/Instruction-Tuning-Survey)
* https://moon-walker.medium.com/%EB%A6%AC%EB%B7%B0-meta-llama%EC%9D%98-%EC%B9%9C%EC%B2%99-stanford-univ%EC%9D%98-alpaca-ec82d432dc25
* https://lmsys.org/blog/2023-03-30-vicuna/
* https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM?tab=readme-ov-file#how-good-is-the-data

## Instruction Tuning - upstage/SOLAR-10.7B-v1.0 (🚀️ it's my first  try)

Objectives

- pre-trained 모델을 대상으로 주어진 문제를 어떻게 접근해야 할지에 대해 instruction 에 🚀️ 이를 finetune
- LoRA 기반으로 parameter efficient finetune 으로 mlp layer (일부 모듈) 에 adapter 를 붙여서 진행

Pretrained Model

- [upstage/SOLAR-10.7B-v1.0 · Hugging Face](https://huggingface.co/upstage/SOLAR-10.7B-v1.0)

Train Data

- [kyujinpy/Open-platypus-Commercial · Datasets at Hugging Face](https://huggingface.co/datasets/kyujinpy/Open-platypus-Commercial)

Result

- [changsun/changsun · Hugging Face](https://huggingface.co/changsun/changsun)

Todo 😕

- fine tune 완료된 모델에 pre-trained 모델 입력처럼 넣었는데 답이 없음 👎

## Instruction Tuning - upstage/SOLAR-10.7B-v.1.0 ( second try )

Objectives

- 이미 받아놓은 모델가지고 다른 데이터 셋을 가지고 instruction tuning 해봄
- GPT - 4 가지고 만든 데이터 (e.g., alphaca) 를 사용

Pretrained Model (상동)

Train Data

- //

Result

- //

Todo

- //
