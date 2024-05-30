from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
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
    "mistralai/Mistral-7B-Instruct-v0.2", device_map='auto', quantization_config=bnb_config
)

model = PeftModel.from_pretrained(model, './output_dir')

tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

model_inputs = tokenizer(
    ["### Human: Hey what can you do for me? .### Assistant:"], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs)
output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(output)
