import torch
from huggingface_hub import login
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams  # Import vLLM

# Login to Hugging Face
login(token='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')



from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    torch_dtype=torch.bfloat16
).to("cuda")  # or .to("cpu")



# Load datasets
import json
log_file = '/home/scur1410/Project/LLM_Results/distil-deepseek-llama8B.log'

train_data = []
with open('/home/scur1410/Project/BBQ/data/Age.jsonl', 'r') as f:
    for line in f:
        train_data.append(json.loads(line))



# **Batching Parameters**
batch_size = 16   # Adjust as needed
start_index = 0


# **Processing in Batches**
for batch_start in range(start_index, len(train_data), batch_size):
    batch_end = min(batch_start + batch_size, len(train_data))
    batch_samples = train_data[batch_start:batch_end]

    batch_input_texts = []
    for sample in batch_samples:
        question = sample['question']
        ans0 = sample['ans0']
        ans1 = sample['ans1']
        ans2 = sample['ans2']
        context = sample['context']

        # Create input text by replacing placeholders
        input_text = question + '\n' + '(a)' + ans0 + '(b)' + ans1 + '(c)' + ans2 + '\n' + context
        batch_input_texts.append(input_text)

    # **Run Model Inference with vLLM**
    inputs = tokenizer(batch_input_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    # Run model inference with the tokenized inputs
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.0001)

    # Decode the generated outputs
    batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    
    # **Logging Results**
    with open(log_file, 'a') as file:
        for i, output in enumerate(batch_outputs):
            index = batch_start + i
            file.write("***************\n")
            file.write(f"Age sample {index} generation STARTED\n")
            file.write(output + "\n")
            file.write(f"Age sample {index} generation FINISHED\n")
            file.write("***************\n")

    print(f"Processed batch {batch_start}-{batch_end - 1}, results saved to {log_file}\n")