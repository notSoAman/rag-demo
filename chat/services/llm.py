from openai import OpenAI
from os import getenv
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

def generate_answer(question: str) -> str:
    completion = client.chat.completions.create(
        model="poolside/laguna-xs-2.1:free",
        messages=[
            {"role": "user", "content": question}
        ],
    )
    return completion.choices[0].message.content