# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline
# import torch
# import os
# import csv
# from datasets import load_dataset

# device = 'mps'  # the device to load the model onto

# model_name = 'ystemsrx/Qwen2-Boundless'

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# messages = [{"role": "system", "content": ""}]

# csv_path = os.path.join(os.path.dirname(__file__), 'emotions.csv')

# f = open(csv_path, mode='w')
# writer = csv.writer(f)

# emotion_analyzer = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student")

# # Mapping emotions to happiness scale
# emotion_happiness_map = {
#     "joy": 0.2,        # Slight happiness boost
#     "love": 0.3,       # More warmth in response
#     "surprise": 0.4,   # Curious/exciting responses
#     "neutral": 0.5,    # Standard response
#     "sadness": 0.7,    # More happiness needed
#     "anger": 0.9,      # Max happiness boost to calm user
#     "fear": 0.8,       # Reassuring happy tone
#     "disgust": 0.8,    # Positive rewording
#     "guilt": 0.7       # Gentle encouragement
# }

# def analyze_emotion(prompt: str) -> float:
#     """
#     Detects the user's emotion from the prompt and returns a happiness scale (0 to 1).
#     """
#     emotions = emotion_analyzer(prompt)
#     top_emotion = emotions[0]['label']  # Get the most probable emotion
    
#     print(f'{top_emotion} | {emotion_happiness_map.get(top_emotion, 0.5)}')

#     # Map detected emotion to a happiness scale (default to 0.5 if unknown)
#     return emotion_happiness_map.get(top_emotion, 0.5)

# def generate_response(prompt: str, happiness: float) -> str:
#     """
#     Generates a response using Qwen2, adjusting happiness based on detected emotion.
#     """
#     prompt += f"Make the response {happiness*100}% happier."
    
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
#     print(f"LLM: {response}")
#     writer.writerow([f"LLM: {response}"])

#     messages.append({"role": "assistant", "content": response})

#     return response

# def compute_reward(user_happiness: float, llm_happiness: float) -> float:
#     """
#     Computes the reward for the user given their happiness and the LLM's happiness.
#     """
#     return llm_happiness - user_happiness

# prompt = input('You: ')
# writer.writerow([f'User: {prompt}'])
# user_happiness = analyze_emotion(prompt)  # Get happiness scale

# for i in range(10):
#     response = generate_response(prompt, user_happiness)
#     writer.writerow([f'System: {response}'])
#     llm_happiness = analyze_emotion(response)  # Get happiness scale
    
#     prompt = input('You: ')
#     writer.writerow([f'User: {prompt}'])
#     user_happiness = analyze_emotion(prompt)  # Get happiness scale
    
#     reward = compute_reward(user_happiness, llm_happiness)
    
#     print(f'LLM Happiness: {llm_happiness} | User Happiness: {user_happiness} | Reward: {reward}')
    
    
import sys
import os
import csv
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel, QHBoxLayout
from PyQt5.QtGui import QTextCursor
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline

class ChatbotUI(QWidget):
    def __init__(self):
        super().__init__()
        
        self.initUI()
        self.setupModels()
        
        self.messages = [{"role": "system", "content": ""}]
        self.csv_path = os.path.join(os.path.dirname(__file__), 'emotions.csv')
        self.csv_file = open(self.csv_path, mode='w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["User Input", "LLM Response", "User Happiness", "LLM Happiness", "Reward"])
    
    def initUI(self):
        self.setWindowTitle("Emotion-Aware Chatbot")
        self.setGeometry(100, 100, 500, 400)
        
        layout = QVBoxLayout()
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)
        
        self.input_box = QLineEdit()
        layout.addWidget(self.input_box)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.processInput)
        layout.addWidget(self.send_button)
        
        self.emotion_label = QLabel("User Emotion: Neutral")
        layout.addWidget(self.emotion_label)
        
        self.reward_label = QLabel("Reward: 0.0")
        layout.addWidget(self.reward_label)
        
        self.setLayout(layout)
    
    def setupModels(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model_name = 'ystemsrx/Qwen2-Boundless'
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.emotion_analyzer = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student")
        
        self.emotion_happiness_map = {
            "joy": 0.2, "love": 0.3, "surprise": 0.4, "neutral": 0.5, 
            "sadness": 0.7, "anger": 0.9, "fear": 0.8, "disgust": 0.8, "guilt": 0.7
        }
    
    def analyzeEmotion(self, text):
        emotions = self.emotion_analyzer(text)
        top_emotion = emotions[0]['label']
        happiness = self.emotion_happiness_map.get(top_emotion, 0.5)
        
        self.emotion_label.setText(f"User Emotion: {top_emotion}")
        return happiness
    
    def generateResponse(self, prompt, happiness):
        prompt += f" Make the response {happiness*100}% happier."
        self.messages.append({"role": "user", "content": prompt})
        
        text = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(model_inputs.input_ids, attention_mask=model_inputs.attention_mask, max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.messages.append({"role": "assistant", "content": response})
        
        return response
    
    def computeReward(self, user_happiness, llm_happiness):
        return llm_happiness - user_happiness
    
    def processInput(self):
        user_input = self.input_box.text().strip()
        if not user_input:
            return
        
        self.chat_display.append(f"<b>User:</b> {user_input}")
        user_happiness = self.analyzeEmotion(user_input)
        response = self.generateResponse(user_input, user_happiness)
        llm_happiness = self.analyzeEmotion(response)
        reward = self.computeReward(user_happiness, llm_happiness)
        self.chat_display.moveCursor(QTextCursor.End)
        self.chat_display.append(f"<b>LLM:</b> {response}")
        
        self.reward_label.setText(f"Reward: {reward:.2f}")
        
        self.writer.writerow([user_input, response, user_happiness, llm_happiness, reward])
        
        self.input_box.clear()
    
    def closeEvent(self, event):
        self.csv_file.close()
        event.accept()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatbotUI()
    window.show()
    sys.exit(app.exec_())