from datasets import load_dataset
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoTokenizer, GenerationConfig, AutoModelForSequenceClassification, EncoderDecoderCache
from trl import AutoModelForSeq2SeqLMWithValueHead
import csv
import numpy as np
import math
import os

class PPO:
    def __init__(self, model, dataset, min_text_length, max_text_length, reward_model):
        self.device = None
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        print('Using device:', self.device)
        
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        
        self.model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
        self.dataset = load_dataset(dataset)
        self.file_path = 'dialogsum'
        
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model).to(self.device)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model)
        
    def preprocess_dataset(self):
        if os.path.exists(self.file_path):
            self.dataset = load_dataset(self.file_path)
            return 
        
        self.dataset = self.dataset.filter(lambda x: len(x['dialogue']) > self.min_text_length and len(x['dialogue']) <= self.max_text_length, batched=False)
        self.dataset = self.dataset.map(self.tokenize, batched=False)
        self.dataset.save_to_disk(self.file_path)
        
    
    def tokenize(self, text):    
        prompt = f"""
        Summarize the following conversation.

        {text['dialogue']}

        Summary:
        """
        text['input_ids'] = self.tokenizer.encode(prompt)
        text['query'] = self.tokenizer.decode(text['input_ids'])
        return text
    
    def reward_function(self, batch_input_ids, batch_attention_mask):
        batch_input_ids = batch_input_ids.to(self.device)
        batch_attention_mask = batch_attention_mask.to(self.device)
        logits = self.reward_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
        probabilities = logits.softmax(dim=-1).tolist()
        non_toxic_rewards = np.array([p[1] for p in probabilities]) # Get the probability of toxic
        rewards = (non_toxic_rewards - 0.9) * 100
        return rewards
    
    def train(self, epochs=10, batch_size=64):
        self.epochs = epochs
        self.batch_size = batch_size
        self.preprocess_dataset()
        self.dataset.set_format(type='torch')
        
        optimizer = optim.Adam(self.model.parameters(),lr=1e-5)
        loss_fn = nn.MSELoss()
        
        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1}/{self.epochs}:')
            epoch_train_rewards = []
            i = 0
            step = 0
            while i < len(self.dataset['train']):
                batch = self.dataset['train'].select(range(i, min(i + self.batch_size, len(self.dataset['train']))))
                batch_tokens = self.tokenizer(batch['query'], return_tensors='pt', padding=True, truncation=True).to(self.device)
                batch_input_ids = batch_tokens.input_ids
                batch_attention_mask = batch_tokens.attention_mask
                
                with torch.no_grad():
                    old_rewards = self.reward_function(batch_input_ids, batch_attention_mask)
                    old_rewards = torch.tensor(old_rewards, dtype=torch.float16, device=self.device, requires_grad=True)
                
                outputs = self.model.generate(batch_input_ids, max_new_tokens=100)
                responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                response_tokens = self.tokenizer(responses, return_tensors='pt', padding=True, truncation=True).to(self.device)
                response_input_ids = response_tokens.input_ids
                response_attention_mask = response_tokens.attention_mask
                
                new_rewards = self.reward_function(response_input_ids, response_attention_mask)
                new_rewards = torch.tensor(new_rewards, dtype=torch.float16, device=self.device, requires_grad=True)
                
                advantages = new_rewards - old_rewards
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = advantages.unsqueeze(1)
                
                decoder_input_ids = batch_input_ids.clone()[:, :-1]
                outputs = self.model(input_ids=batch_input_ids, decoder_input_ids=decoder_input_ids)
                logits = outputs[0][:, :-1, :]
                log_probs = torch.log_softmax(logits, dim=-1)

                next_token_ids = batch_input_ids[:, 1:1:log_probs.shape[1] + 1]
                log_probs = torch.gather(log_probs, dim=-1, index=next_token_ids.unsqueeze(-1)).squeeze(-1)
                
                with torch.no_grad():
                    old_log_probs = log_probs.detach()
                
                # log_probs = torch.log_softmax(self.model(batch_input_ids).logits, dim=-1)

                ratio = torch.exp(log_probs - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                
                value_loss = loss_fn(new_rewards, old_rewards)

                loss = policy_loss + 0.5 * value_loss
                
                loss = loss_fn(new_rewards.to(self.device), old_rewards.to(self.device))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                mean_rewards = new_rewards.mean().item()
                epoch_train_rewards.append(mean_rewards)
                i += self.batch_size
                step += 1
                print(f'\tStep {step}/{int(math.ceil(len(self.dataset['train']) / self.batch_size))} | Reward: {mean_rewards}')
            print(f'Avg Training Reward: {np.mean(epoch_train_rewards)}')
            
            with torch.no_grad():
                val_tokens = self.tokenizer(self.dataset['validation']['dialogue'], return_tensors='pt', padding=True, truncation=True).to(self.device)
                val_input_ids = val_tokens.input_ids
                val_attention_mask = val_tokens.attention_mask
                epoch_val_rewards = self.reward_function(val_input_ids, val_attention_mask)
            print(f'Avg Validation Reward: {np.mean(epoch_val_rewards)}')

 
dataset = 'knkarthick/dialogsum'
model = 'google/flan-t5-small'
toxicity_model = 'facebook/roberta-hate-speech-dynabench-r4-target'

epochs = 10
batch_size = 32
min_text_length = 100
max_text_length = 1000

generation_config = GenerationConfig(
    max_new_tokens=10000,  # Increase max tokens
    do_sample=True,      # Enable sampling for diversity
    temperature=0.9,     # Adjust randomness
    top_p=0.95,          # Nucleus sampling for diverse outputs
    top_k=250            # Limit to top 50 tokens for quality
)

agent = PPO(model=model,
            dataset=dataset,
            min_text_length=min_text_length,
            max_text_length=max_text_length,
            reward_model=toxicity_model,
        )

agent.train(epochs=epochs, batch_size=batch_size)

# print(tokenizer.decode(llm.generate(tokens, generation_config=generation_config)[0], skip_special_tokens=True))

# with open('dialogsum.csv', 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow([dataset[0]['dialogue'], dataset[0]['summary']])
#     writer.writerow([dataset[1]['dialogue'], dataset[1]['summary']])