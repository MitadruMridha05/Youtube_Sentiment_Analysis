import os
import sys
import logging
from typing import Optional, TextIO

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# Logging Configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.INFO)
logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

file_handler = logging.FileHandler("data_ingestion.log")
file_handler.setLevel(logging.INFO)


