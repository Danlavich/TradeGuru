from openai import OpenAI

api_key = "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"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


prompt = ("You are an expert broker. You have to make predictions how the related news affect stock prices of given companies."
          "You will be given the company's ticker and economy news that may affect the company's stock prices."
          "Your task is to work with the value of stock rating, which has to be in range of -1 to 1,"
          "where -1 is absolute fall of stock price, 0 means the price remains exactly the same and 1 means greatest possible growth of stocks price."
          "You will be given the latest value of stock rating and you have to decide how the given news would affect it."
          "If you are not given the latest value, you have to decide what it is from scratch. Your response has to contain only the result value."
          "The ticker of the company, stock price of which you are going to work with: "
          )
#Передаем нужный тикер
ticker = "AAPL"
prompt += ticker

#to do добавить уже существующее велью

prompt += ". The following text is the news log that you have to work with. "
#Добавляем новостные данные из текстовика
with open("queries.txt", 'r') as file:
    prompt += file.read()



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
