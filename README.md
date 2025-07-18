# BIKE-SHARING-DEMAND-PREDICTION

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = tk.Tk()
root.title("Real-Time Bike Demand Predictor")
root.geometry("420x500")
root.configure(bg="#F0F8FF")

theta = None
dataset_df = None  # Store the DataFrame

# Load CSV and train the model
def load_csv_and_train():
    global theta, dataset_df
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        df = pd.read_csv(file_path)
        dataset_df = df  # Store for graph

        if not all(col in df.columns for col in ['day', 'hour', 'temperature', 'weather', 'demand']):
            messagebox.showerror("Error", "CSV must have columns: day, hour, temperature, weather, demand")
            return

        X = df[['day', 'hour', 'temperature', 'weather']].values
        y = df['demand'].values
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        messagebox.showinfo("Success", "Model trained successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"Training failed:\n{e}")

# Predict demand from user input
def predict():
    try:
        if theta is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return

        day = int(entry_day.get())
        hour = int(entry_hour.get())
        temp = float(entry_temp.get())
        weather = int(entry_weather.get())

        input_x = np.array([1, day, hour, temp, weather])
        predicted_demand = input_x.dot(theta)

        result_label.config(text=f"Predicted Bike Demand: {int(predicted_demand)} 🚲")

    except Exception as e:
        messagebox.showerror("Error", "Invalid input! Please check values.")

# Show graph of actual vs predicted
def show_graph():
    if dataset_df is None or theta is None:
        messagebox.showwarning("Warning", "Please train the model first!")
        return

    try:
        X = dataset_df[['day', 'hour', 'temperature', 'weather']].values
        y_actual = dataset_df['demand'].values
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y_predicted = X_b.dot(theta)

        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(y_actual, label='Actual Demand', marker='o')
        plt.plot(y_predicted, label='Predicted Demand', linestyle='--', marker='x')
        plt.title('Actual vs Predicted Bike Demand')
        plt.xlabel('Entry Index')
        plt.ylabel('Bike Demand')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Error plotting graph:\n{e}")

# GUI Layout
tk.Button(root, text="📁 Upload CSV & Train", command=load_csv_and_train, bg="#007B5E", fg="white").pack(pady=10)

tk.Label(root, text="Enter Day (1=Mon, 7=Sun):", bg="#F0F8FF").pack()
entry_day = tk.Entry(root)
entry_day.pack()

tk.Label(root, text="Enter Hour (0–23):", bg="#F0F8FF").pack()
entry_hour = tk.Entry(root)
entry_hour.pack()

tk.Label(root, text="Temperature (°C):", bg="#F0F8FF").pack()
entry_temp = tk.Entry(root)
entry_temp.pack()

tk.Label(root, text="Weather (1=Clear, 2=Cloudy, 3=Rainy):", bg="#F0F8FF").pack()
entry_weather = tk.Entry(root)
entry_weather.pack()

tk.Button(root, text="🔮 Predict Demand", command=predict, bg="#4682B4", fg="white").pack(pady=10)
tk.Button(root, text="📊 Show Graph", command=show_graph, bg="#6A5ACD", fg="white").pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 14), bg="#F0F8FF")
result_label.pack()

root.mainloop()
