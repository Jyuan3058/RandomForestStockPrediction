# logger.py
import logging
import sys

# Create a custom logger
logger = logging.getLogger("RandomForestStockPrediction")
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logs

# Create handlers
c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler('app.log')
c_handler.setLevel(logging.INFO)  # Console handler set to INFO
f_handler.setLevel(logging.DEBUG)  # File handler set to DEBUG

# Create formatters and add them to handlers
c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)
