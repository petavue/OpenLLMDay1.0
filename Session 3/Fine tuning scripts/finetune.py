from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=bnb_config,
    device_map="auto",
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
       'k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'
    ],
)

model.config.use_cache = False
model = get_peft_model(model, peft_config)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token

training_arguments = TrainingArguments(
    output_dir="./results_latest",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim='paged_adamw_32bit',
    num_train_epochs=8,
    #save_steps=20,
    fp16=True,
    logging_steps=10,
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
)

dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")

def formatting_prompts(example):
    output_text = []
    for i in range(len(example['sql_prompt'])):
        text = f"### NL Question: {example['sql_prompt'][i]}\n ### Context: {example['sql_context'][i]}\n ### Answer: {example['sql'][i]}\n ### Explanation: {example['sql_explanation'][i]}"
        output_text.append(text)
    return output_text


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()

model.save_pretrained("output_dir")
