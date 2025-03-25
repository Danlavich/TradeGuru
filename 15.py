import pandas as pd
import tkinter as tk
from tkinter import ttk

# Фильтрация данных
def apply_filter():
    min_price = float(price_min_entry.get()) if price_min_entry.get() else float('-inf')
    max_price = float(price_max_entry.get()) if price_max_entry.get() else float('inf')
    min_pe = float(pe_min_entry.get()) if pe_min_entry.get() else float('-inf')
    max_pe = float(pe_max_entry.get()) if pe_max_entry.get() else float('inf')

    filtered_df = df[
        (df["Цена"] >= min_price) & (df["Цена"] <= max_price) &
        (df["P/E"] >= min_pe) & (df["P/E"] <= max_pe)
    ]

    for row in table.get_children():
        table.delete(row)

    for index, row in filtered_df.iterrows():
        table.insert("", "end", values=row.tolist())

# Создание интерфейса
root = tk.Tk()
root.title("Фильтр данных")

frame = tk.Frame(root)
frame.pack(pady=20)

tk.Label(frame, text="Цена от:").grid(row=0, column=0)
price_min_entry = tk.Entry(frame)
price_min_entry.grid(row=0, column=1)

tk.Label(frame, text="до:").grid(row=0, column=2)
price_max_entry = tk.Entry(frame)
price_max_entry.grid(row=0, column=3)

tk.Label(frame, text="P/E от:").grid(row=1, column=0)
pe_min_entry = tk.Entry(frame)
pe_min_entry.grid(row=1, column=1)

tk.Label(frame, text="до:").grid(row=1, column=2)
pe_max_entry = tk.Entry(frame)
pe_max_entry.grid(row=1, column=3)

apply_button = tk.Button(frame, text="Применить фильтр", command=apply_filter)
apply_button.grid(row=2, column=0, columnspan=4, pady=10)

table = ttk.Treeview(root, columns=list(df.columns), show="headings")
for col in df.columns:
    table.heading(col, text=col)
    table.column(col, width=100)
table.pack()

apply_filter()  # Отображение всех данных при старте

root.mainloop()
