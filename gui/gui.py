# gui/gui.py

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import traceback
from predictor.price_predictor import PricePredictor
from logger import logger  # Import the configured logger

class PricePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Predictor")
        self.root.geometry("500x400")
        logger.info("Initializing the GUI.")
        self.create_widgets()
        
        # Create a queue to communicate with the thread
        self.queue = queue.Queue()
        
        # Start the periodic check of the queue
        self.root.after(100, self.process_queue)

    def create_widgets(self):
        logger.debug("Creating GUI widgets.")
        # Ticker Input
        ticker_label = ttk.Label(self.root, text="Enter Company Ticker:")
        ticker_label.pack(pady=10)

        self.ticker_entry = ttk.Entry(self.root, width=30)
        self.ticker_entry.pack(pady=5)

        # Predict Button
        predict_button = ttk.Button(self.root, text="Predict", command=self.start_prediction_thread)
        predict_button.pack(pady=20)

        # Results Display
        self.results_text = tk.Text(self.root, height=10, width=60, wrap='word')
        self.results_text.pack(pady=10)

    def start_prediction_thread(self):
        ticker = self.ticker_entry.get().strip().upper()
        logger.info(f"Predict button clicked with ticker: {ticker}")
        if not ticker:
            logger.warning("No ticker entered.")
            messagebox.showwarning("Input Error", "Please enter a valid ticker symbol.")
            return

        # Disable the button to prevent multiple clicks
        logger.debug("Disabling widgets to prevent multiple predictions.")
        self.set_widgets_state('disabled')

        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Running prediction for {ticker}...\n")
        logger.debug("Cleared previous results in the GUI.")

        # Start the prediction in a new thread
        logger.info("Starting prediction thread.")
        thread = threading.Thread(target=self.run_prediction, args=(ticker,), daemon=True)
        thread.start()

    def run_prediction(self, ticker):
        logger.debug(f"Background thread started for ticker: {ticker}")
        try:
            predictor = PricePredictor(ticker)
            logger.info(f"PricePredictor instantiated for ticker: {ticker}")
            results = predictor.run()
            logger.debug(f"Prediction results: {results}")

            # Prepare result message
            result_message = (
                f"Initial Precision Score: {results['initial_precision']:.4f}\n"
                f"Backtest Precision Score: {results['backtest_precision']:.4f}\n"
                f"Prediction Counts: {results['prediction_counts']}\n"
            )
            # Put the result in the queue
            self.queue.put(("success", result_message))
            logger.info(f"Prediction completed for ticker: {ticker}")
        except Exception as e:
            # Capture the traceback
            tb = traceback.format_exc()
            error_message = f"An error occurred:\n{e}\n\n{tb}"
            # Put the error in the queue
            self.queue.put(("error", error_message))
            logger.error(f"Error during prediction for ticker {ticker}: {e}\n{tb}")

    # gui/gui.py
    

    def process_queue(self):
        logger.debug("Processing the message queue.")
        try:
            while True:
                status, message = self.queue.get_nowait()
                logger.debug(f"Retrieved message from queue: status={status}, message={message}")
                if status == "success":
                    logger.debug(f"Inserting message into Text widget: {repr(message)}")
                    self.results_text.insert(tk.END, message)
                    self.results_text.see(tk.END)  # Scroll to the end
                    self.results_text.update_idletasks()  # Force update
                    logger.info("Successfully inserted prediction results into the GUI.")
                elif status == "error":
                    logger.debug(f"Inserting error message into Text widget: {message}")
                    self.results_text.insert(tk.END, f"{message}\n")
                    messagebox.showerror("Error", message)
                    logger.error("Displayed error message to the user.")
                # Re-enable the widgets after processing
                self.set_widgets_state('normal')
                logger.debug("Re-enabled widgets after processing the queue.")
        except queue.Empty:
            logger.debug("No messages in the queue.")
            pass
        # Schedule the next queue check
        self.root.after(100, self.process_queue)


    def set_widgets_state(self, state):
        logger.debug(f"Setting widgets state to {state}.")
        for widget in self.root.winfo_children():
            if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Label, tk.Text)):
                # Check if the widget state is already 'normal'
                if widget.cget("state") == "normal":
                    logger.debug(f"Widget {widget} is already in 'normal' state.")
                else:
                    widget.configure(state=state)

