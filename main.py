# main.py
import tkinter as tk
from gui import PricePredictorGUI  # Import from gui/__init__.py
from logger import logger  # Import the configured logger

def main():
    logger.info("Starting the Stock Price Predictor application.")
    root = tk.Tk()
    app = PricePredictorGUI(root)
    logger.debug("Entering the main loop of the GUI.")
    root.mainloop()
    logger.info("Application has been closed.")

if __name__ == "__main__":
    main()
