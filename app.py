# read env key for openai
from dotenv import load_dotenv
# import OpenAI client
from openai import OpenAI
# to read pdf file and extract text
from pypdf import PdfReader
# to make the UI
import gradio as gr
from pathlib import Path

load_dotenv(override=True)
# Create an instance of the OpenAI class
openai = OpenAI()

BASE_DIR = Path(__file__).parent

reader = PdfReader(BASE_DIR / "me" / "Profile.pdf")

linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

# Read the summary text
with open(BASE_DIR / "me" / "summary.txt", encoding="utf-8") as f:
    summary = f.read()

name = "Ayushi Saxena"
system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s career, background, skills and experience. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer, say so."
system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

def chat(message, history):
    messages = [{'role': 'system', 'content': system_prompt}] + history + [{'role': 'user', 'content': message}]
    response = openai.chat.completions.create(model = "gpt-4o-mini", messages = messages)
    return response.choices[0].message.content

gr.ChatInterface(chat).launch()