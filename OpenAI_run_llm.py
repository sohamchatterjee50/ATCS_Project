from openai import OpenAI
import json
import time

# API Configuration
client = OpenAI(
    api_key="xxx",
    base_url="xxx"
)

def generate_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0001,
        max_tokens=512,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )
    return response.choices[0].message.content.strip()

# Data and Log Configuration
data_path = "/Users/haha/Downloads/Gender_identity.jsonl"
log_file = "gender_identity_results.log"

# Load Data
train_data = []
with open(data_path, 'r') as f:
    for line in f:
        train_data.append(json.loads(line))

batch_size = 16
start_index = 0

for batch_start in range(start_index, len(train_data), batch_size):
    batch_end = min(batch_start + batch_size, len(train_data))
    batch_samples = train_data[batch_start:batch_end]

    for idx, sample in enumerate(batch_samples):
        question = sample['question']
        ans0 = sample['ans0']
        ans1 = sample['ans1']
        ans2 = sample['ans2']
        context = sample['context']

        prompt = (
            f"Question: {question}\n"
            f"(a) {ans0}\n"
            f"(b) {ans1}\n"
            f"(c) {ans2}\n"
            f"Context: {context}\n\n"
            "Please choose the most appropriate answer by only stating (a), (b), or (c), "
            "and then explain your reasoning clearly."
        )

        try:
            response_content = generate_response(prompt)
        except Exception as e:
            response_content = f"Error: {str(e)}"

        with open(log_file, 'a') as file:
            file.write("***************\n")
            file.write(f"Gender sample {batch_start + idx} generation STARTED\n")
            file.write(response_content + "\n")
            file.write(f"Gender sample {batch_start + idx} generation FINISHED\n")
            file.write("***************\n")

        print(f"Sample {batch_start + idx} done.")
        time.sleep(1)
