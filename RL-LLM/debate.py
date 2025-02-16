from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import csv
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset

device = 'mps'  # the device to load the model onto

model_name = 'ystemsrx/Qwen2-Boundless'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

reward_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(device)
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

def compute_reward(response):
    batch_inputs = reward_tokenizer(response, return_tensors="pt", truncation=True, padding=True).to(device)
    logits = self.reward_model(input_ids=batch_inputs.input_ids, attention_mask=batch_inputs.attention_mask).logits
    probabilities = logits.softmax(dim=-1).tolist()
    with torch.no_grad():
        logits = reward_model(**inputs).logits
    non_toxic_rewards = np.array([p[1] for p in probabilities]) # Get the probability of toxic
    rewards = (non_toxic_rewards - 0.9) * 100
    return rewards

config = PPOConfig(
    batch_size=32,
    learning_rate=1e-5,
    output_dir='output'
    )

# ppo_trainer = PPOTrainer(config, model, tokenizer)

messages = [
    {"role": "system", "content": ""}
] * 5

# csv_path = os.path.join(os.path.dirname(__file__), 'debate.csv')

# f = open(csv_path, mode='w')
# writer = csv.writer(f)

def tokenize(text):    
        prompt = f"""
        Summarize the following conversation.

        {text['dialogue']}

        Summary:
        """
        text['input_ids'] = tokenizer.encode(prompt)
        text['query'] = tokenizer.decode(text['input_ids'])
        return text
    
def preprocess_dataset(dataset, file_path):
        if os.path.exists(file_path):
            dataset = load_dataset(file_path)
            return dataset
        
        dataset = dataset.filter(lambda x: len(x['dialogue']) > 100 and len(x['dialogue']) <= 1000, batched=False)
        dataset = dataset.map(tokenize, batched=False)
        dataset.save_to_disk(file_path)
        
        return dataset

file_path = 'dialogsum'
dataset = load_dataset('knkarthick/dialogsum')
dataset = preprocess_dataset(dataset, file_path)

# prompt = "Hi, how are you?"
# print(f'Initial prompt: {prompt}')
# writer.writerow([f'Initial prompt: {prompt}'])

for epoch in range(5):
    print(f'Epoch {epoch+1}/5:')
    i = 0
    step = 1
    while i < len(dataset['train']):
        batch = dataset['train'].select(range(i, min(i + 32, len(dataset['train']))))
        batch_tokens = tokenizer(batch['query'], return_tensors='pt', padding=True, truncation=True).to(device)
        batch_input_ids = batch_tokens.input_ids.to(device)
        batch_attention_mask = batch_tokens.attention_mask.to(device)
        response = model.generate(batch_input_ids, attention_mask=batch_attention_mask, max_new_tokens=512)
        
        with torch.no_grad():
            rewards = compute_reward(response)
        
        print(f'Step: {step} | Reward: {np.mean(rewards)}')
        
        ppo_trainer.add_batch(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            rewards=rewards
        )
        
        if len(ppo_trainer.rewards) >= 32:
            ppo_trainer.train_step()
            ppo_trainer.clear_batch()
            
        i += 32
        step += 1

# while True:
#     messages.extend([{"role": "user", "content": prompt}] * 5)

#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
    
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)

#     for i in range(5):
#         generated_ids = model.generate(
#             model_inputs.input_ids,
#             attention_mask=model_inputs.attention_mask,
#             max_new_tokens=512
#         )
        
#         generated_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#         ]

#         response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#         print(f"LLM 1: {response}")
#         # writer.writerow([f"LLM 1: {response}"])

#         messages.append({"role": "assistant", "content": response})

# while True:
#     messages.append({"role": "user", "content": prompt})

#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
    
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)

#     generated_ids = model.generate(
#         model_inputs.input_ids,
#         attention_mask=model_inputs.attention_mask,
#         max_new_tokens=512
#     )
    
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     print(f"LLM 1: {response}")
#     writer.writerow([f"LLM 1: {response}"])

#     messages.append({"role": "assistant", "content": response})
    
#     prompt = response
    
#     messages.append({"role": "user", "content": prompt})

#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
    
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)

#     generated_ids = model.generate(
#         model_inputs.input_ids,
#         attention_mask=model_inputs.attention_mask,
#         max_new_tokens=512
#     )
    
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     print(f"LLM 2: {response}")
#     writer.writerow([f"LLM 2: {response}"])

#     messages.append({"role": "assistant", "content": response})
    
#     prompt = response
