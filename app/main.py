from typing import Union
from dotenv import load_dotenv
from fastapi import FastAPI
import os
import openai
import base64

load_dotenv()

LLM_PROXY_SERVICE_API_URL = os.getenv('LLM_PROXY_SERVICE_API_URL')
LLM_PROXY_SERVICE_API_KEY = os.getenv('LLM_PROXY_SERVICE_API_KEY')

app = FastAPI()


# 🧩 Optional helper to encode images (for multimodal models)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@app.post("/generate")
def generate_answer(q: Union[str, None] = None):
    # 🚀 Initialize the client
    client = openai.AzureOpenAI(
        api_key=LLM_PROXY_SERVICE_API_KEY,  # Your team api_key
        azure_endpoint=LLM_PROXY_SERVICE_API_URL,
        api_version="2024-02-01"
    )
    # 💬 Example text request
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": q}
        ]
    )

    print(response)

    return {"response": response.choices[0].message.content}
