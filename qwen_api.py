from openai import OpenAI

api_key = "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"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

with open("queries.txt", 'r') as file:
    content = file.read()


prompt = content

response = client.chat.completions.create(
    model="qwen/qwen2.5-vl-72b-instruct:free",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
)


print(response.choices[0].message.content)
