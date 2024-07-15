import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.metrics import r2_score
from prophet import Prophet

def forecast_temperature(csv_file, param_col, pred_time):
    
    pred_time_ = pd.to_datetime(pred_time)
    freq = 'min' # to be changed to desired frequency
    
    df_macro = pd.read_csv(csv_file, header=0, parse_dates=[0], date_format='%Y-%m-%d %H:%M:%S')

    # TESTING

    date_col = df_macro.columns[0]
    n_preds = int((pred_time_ - pd.to_datetime(df_macro[date_col].iloc[-1]))/
                  (pd.to_datetime(df_macro[date_col].iloc[-1]) - pd.to_datetime(df_macro[date_col].iloc[-2])))

    df = df_macro[[date_col, param_col]]
    df.columns = ['ds', 'y']

    train = df.iloc[:-n_preds]
    test = df.iloc[-n_preds:]

    m = Prophet()
    m.fit(train)
    print(n_preds)
    print(pred_time_)
    print(pd.to_datetime(df_macro[date_col].iloc[-1]))
    print(pd.to_datetime(df_macro[date_col].iloc[-2]))
    future = m.make_future_dataframe(n_preds, freq=freq)
    entire_forecast = m.predict(future)

    predictions = entire_forecast.iloc[-n_preds:][['ds', 'yhat']]
    accuracy = r2_score(predictions['yhat'], test['y']) * 100

    # PREDICTING

    m_ = Prophet()
    m_.fit(df)
    future_ = m_.make_future_dataframe(n_preds, freq=freq)
    entire_forecast_ = m_.predict(future_)

    predictions_ = entire_forecast_.iloc[-n_preds:][['ds', 'yhat']]

    # PLOTTING
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(predictions_['ds'], predictions_['yhat'])
    ax.set_title('Accuracy: ' + str(accuracy) + '%')
    ax.set_xlabel("Date")
    ax.set_ylabel(param_col)
    predictions_.columns = ["Date", "Predicted parameter"]
    return predictions_, fig, accuracy

def display_results(data, figure, accuracy):
    # Clear previous results if any
    for widget in result_frame.winfo_children():
        widget.destroy()

    columns = list(data.columns)

    # Create table widget with dynamic columns
    table = ttk.Treeview(result_frame, columns=columns, show="headings")
    table.heading("#0", text="Index")

    # Set headings for all columns (except index)
    for col in columns:
        table.heading(col, text=col, anchor=tk.W)
    
    # Insert data from DataFrame
    for index, row in data.iterrows():
        table.insert("", tk.END, values=list(row))

    table.grid(row=1, column=0, sticky=tk.NSEW)  # Grid table in result frame (adjust if needed)

    # Create vertical scrollbar
    scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=table.yview)
    scrollbar.grid(row=1, column=1, sticky=tk.NS)
    table.configure(yscrollcommand=scrollbar.set)

    # Create matplotlib canvas
    canvas = FigureCanvasTkAgg(figure, result_frame)
    canvas.get_tk_widget().grid(row=1, column=2, sticky=tk.NSEW)  # Grid canvas in result frame

    # Configure column and row weights to make the layout responsive
    result_frame.grid_columnconfigure(0, weight=1)
    result_frame.grid_columnconfigure(2, weight=1)
    result_frame.grid_rowconfigure(1, weight=1)


def show_additional_inputs():
    # Display additional inputs
    
    global parameter_entry
    parameters = pd.read_csv(db_file_entry.get()).columns.tolist()[1:]
    parameter_label = tk.Label(input_frame, text="Select parameter to predict: ")
    parameter_entry = tk.OptionMenu(input_frame, clicked, *parameters)
    parameter_label.grid(row=1, column=0, sticky=tk.W, pady=5)
    parameter_entry.grid(row=1, column=1, sticky=tk.W)
    datetime_label.grid(row=2, column=0, sticky=tk.W, pady=5)
    datetime_entry.grid(row=2, column=1, sticky=tk.EW)
    submit_button.grid(row=3, column=0, columnspan=2, pady=10)
    initial_submit_button.grid_remove()

# Main window
window = tk.Tk()
window.title("Weather Forecasting Model")

# Center-align all content within the window
window.grid_columnconfigure(0, weight=1)

# Label for title
title_label = tk.Label(window, text="Weather Forecasting Model", font=("Arial", 16))
title_label.grid(row=0, column=0, columnspan=2, pady=10)  # Grid title label

# Input frame
input_frame = tk.Frame(window)
input_frame.grid(row=1, column=0, columnspan=2, pady=10)  # Grid input frame

# Database file label and entry
db_file_label = tk.Label(input_frame, text="Enter database file: ")
db_file_label.grid(row=0, column=0, sticky=tk.W)  # Grid label with west anchor

db_file_entry = tk.Entry(input_frame)
db_file_entry.grid(row=0, column=1, sticky=tk.EW)  # Grid entry with east and west anchor

clicked = tk.StringVar()
clicked.set("Select")

# Initial submit button
initial_submit_button = tk.Button(input_frame, text="Submit", command=show_additional_inputs)
initial_submit_button.grid(row=0, column=2, padx=5)  # Grid submit button next to the entry

# Date & time label and entry (hidden initially)
datetime_label = tk.Label(input_frame, text="Enter date & time (YYYY/MM/DD HH:MM:SS): ")
datetime_entry = tk.Entry(input_frame)

submit_button = tk.Button(window, text="Submit", command=lambda: 
                          display_results(*forecast_temperature(db_file_entry.get(), 
                                                                  clicked.get(), 
                                                                  datetime_entry.get())))

# Result frame
result_frame = tk.Frame(window)
result_frame.grid(row=4, column=0, columnspan=2)  # Grid result frame

# Run the application
window.mainloop()
