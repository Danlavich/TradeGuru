from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Загрузка модели
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# Загрузка процессора
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")



def process_query(query):
    messages = [{
        "role": "user",
        "content": [{"text": query.strip()}]  # Явное указание текстового контента
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,  # Убрать список []
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Добавить параметры генерации
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9
    )

    return processor.decode(generated_ids[0], skip_special_tokens=True)


# Обработка всех запросов и запись результатов
results = []
for query in queries:
    response = process_query(query)
    results.append(f"Запрос: {query.strip()}\nОтвет: {response}\n")

# Сохранение результатов в файл
with open("responses.txt", "w", encoding="utf-8") as file:
    file.writelines(results)
