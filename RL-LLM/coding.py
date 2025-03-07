# import sys
# import os
# import csv
# import torch
# from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel, QHBoxLayout
# from PyQt5.QtGui import QTextCursor
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline

# class ChatbotUI(QWidget):
#     def __init__(self):
#         super().__init__()
        
#         self.initUI()
#         self.setupModels()
        
#         self.messages = [{"role": "system", "content": ""}]
#         self.csv_path = os.path.join(os.path.dirname(__file__), 'emotions.csv')
#         self.csv_file = open(self.csv_path, mode='w', newline='')
#         self.writer = csv.writer(self.csv_file)
#         self.writer.writerow(["User Input", "LLM Response", "User Happiness", "LLM Happiness", "Reward"])
    
#     def initUI(self):
#         self.setWindowTitle("Emotion-Aware Chatbot")
#         self.setGeometry(100, 100, 500, 400)
        
#         layout = QVBoxLayout()
        
#         self.chat_display = QTextEdit()
#         self.chat_display.setReadOnly(True)
#         layout.addWidget(self.chat_display)
        
#         self.input_box = QLineEdit()
#         layout.addWidget(self.input_box)
        
#         self.send_button = QPushButton("Send")
#         self.send_button.clicked.connect(self.processInput)
#         layout.addWidget(self.send_button)
        
#         self.emotion_label = QLabel("User Emotion: Neutral")
#         layout.addWidget(self.emotion_label)
        
#         self.reward_label = QLabel("Reward: 0.0")
#         layout.addWidget(self.reward_label)
        
#         self.setLayout(layout)
    
#     def setupModels(self):
#         self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
#         self.model_name = 'ystemsrx/Qwen2-Boundless'
#         self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto")
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.emotion_analyzer = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student")
        
#         self.emotion_happiness_map = {
#             "joy": 0.2, "love": 0.3, "surprise": 0.4, "neutral": 0.5, 
#             "sadness": 0.7, "anger": 0.9, "fear": 0.8, "disgust": 0.8, "guilt": 0.7
#         }
    
#     def analyzeEmotion(self, text):
#         emotions = self.emotion_analyzer(text)
#         top_emotion = emotions[0]['label']
#         happiness = self.emotion_happiness_map.get(top_emotion, 0.5)
        
#         self.emotion_label.setText(f"User Emotion: {top_emotion}")
#         return happiness
    
#     def generateResponse(self, prompt, happiness):
#         prompt += f" Make the response {happiness*100}% happier."
#         self.messages.append({"role": "user", "content": prompt})
        
#         text = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
#         model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
#         generated_ids = self.model.generate(model_inputs.input_ids, attention_mask=model_inputs.attention_mask, max_new_tokens=512)
#         generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        
#         response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#         self.messages.append({"role": "assistant", "content": response})
        
#         return response
    
#     def computeReward(self, user_happiness, llm_happiness):
#         return llm_happiness - user_happiness
    
#     def processInput(self):
#         user_input = self.input_box.text().strip()
#         if not user_input:
#             return
        
#         self.chat_display.append(f"<b>User:</b> {user_input}")
#         user_happiness = self.analyzeEmotion(user_input)
#         response = self.generateResponse(user_input, user_happiness)
#         llm_happiness = self.analyzeEmotion(response)
#         reward = self.computeReward(user_happiness, llm_happiness)
#         self.chat_display.moveCursor(QTextCursor.End)
#         self.chat_display.append(f"<b>LLM:</b> {response}")
        
#         self.reward_label.setText(f"Reward: {reward:.2f}")
        
#         self.writer.writerow([user_input, response, user_happiness, llm_happiness, reward])
        
#         self.input_box.clear()
    
#     def closeEvent(self, event):
#         self.csv_file.close()
#         event.accept()
        
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = ChatbotUI()
#     window.show()
#     sys.exit(app.exec_())

import sys
import os
import csv
import torch
import subprocess
import re
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel, QHBoxLayout
from PyQt5.QtGui import QTextCursor
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model_name = 'ystemsrx/Qwen2-Boundless'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = []
new_py_file = 'temp.py'

def extract_code(response):
    parts = response.split("```")
    if len(parts) > 1:
        return parts[1].strip()
    return None

prompt = 'Give me the python code to print the first 100 integers'
messages.append({"role": "user", "content": prompt})

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
print('LLM:', response)

code = extract_code(response)

with open(new_py_file, 'w') as file:
    file.write(code)

print(f'File {new_py_file} created successfully...')
print(f'Running {new_py_file}...')
result = subprocess.run(['python', new_py_file], capture_output=True, text=True)

print(f'Output from {new_py_file}:')
print(result.stdout)
print('Errors (if any):')
print(result.stderr)