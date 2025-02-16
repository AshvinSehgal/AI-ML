from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import os
import csv
from datasets import load_dataset

device = 'mps'  # the device to load the model onto

model_name = 'ystemsrx/Qwen2-Boundless'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# reward_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
# reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(device)
# reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

# def compute_reward(response):
#     batch_inputs = reward_tokenizer(response, return_tensors="pt", truncation=True, padding=True).to(device)
#     logits = self.reward_model(input_ids=batch_inputs.input_ids, attention_mask=batch_inputs.attention_mask).logits
#     probabilities = logits.softmax(dim=-1).tolist()
#     with torch.no_grad():
#         logits = reward_model(**inputs).logits
#     non_toxic_rewards = np.array([p[1] for p in probabilities]) # Get the probability of toxic
#     rewards = (non_toxic_rewards - 0.9) * 100
#     return rewards

messages = [{"role": "system", "content": ""}]

csv_path = os.path.join(os.path.dirname(__file__), 'emotions.csv')

f = open(csv_path, mode='w')
writer = csv.writer(f)

# def tokenize(text):    
#     prompt = f"""
#     Summarize the following conversation.

#     {text['dialogue']}

#     Summary:
#     """
#     text['input_ids'] = tokenizer.encode(prompt)
#     text['query'] = tokenizer.decode(text['input_ids'])
#     return text
    
# def preprocess_dataset(dataset, file_path):
#     if os.path.exists(file_path):
#         dataset = load_dataset(file_path)
#         return dataset
    
#     dataset = dataset.filter(lambda x: len(x['dialogue']) > 100 and len(x['dialogue']) <= 1000, batched=False)
#     dataset = dataset.map(tokenize, batched=False)
#     dataset.save_to_disk(file_path)
    
#     return dataset

# file_path = 'emotions'
# dataset = load_dataset('knkarthick/dialogsum')
# dataset = preprocess_dataset(dataset, file_path)

# prompt = "Hi, how are you?"
# print(f'You: {prompt}')

emotion_analyzer = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student")

# Mapping emotions to happiness scale
emotion_happiness_map = {
    "joy": 0.2,        # Slight happiness boost
    "love": 0.3,       # More warmth in response
    "surprise": 0.4,   # Curious/exciting responses
    "neutral": 0.5,    # Standard response
    "sadness": 0.7,    # More happiness needed
    "anger": 0.9,      # Max happiness boost to calm user
    "fear": 0.8,       # Reassuring happy tone
    "disgust": 0.8,    # Positive rewording
    "guilt": 0.7       # Gentle encouragement
}

def analyze_emotion(prompt: str) -> float:
    """
    Detects the user's emotion from the prompt and returns a happiness scale (0 to 1).
    """
    emotions = emotion_analyzer(prompt)
    top_emotion = emotions[0]['label']  # Get the most probable emotion

    # Map detected emotion to a happiness scale (default to 0.5 if unknown)
    return emotion_happiness_map.get(top_emotion, 0.5)

def generate_response(prompt: str) -> str:
    """
    Generates a response using FLAN-T5, adjusting happiness based on detected emotion.
    """
    happiness = analyze_emotion(prompt)  # Get happiness scale
    
    prompt += f"Make the response {happiness*100}% happier."
    
    messages.append({"role": "user", "content": prompt})
    
    print(f'LLM is {happiness * 100}% happy!')
    writer.writerow([f'LLM is {happiness * 100}% happy!'])
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=512
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"LLM: {response}")
    writer.writerow([f"LLM: {response}"])

    messages.append({"role": "assistant", "content": response})

    return response

for i in range(10):
    prompt = input('You: ')
    writer.writerow([f'User: {prompt}'])
    
    response = generate_response(prompt)