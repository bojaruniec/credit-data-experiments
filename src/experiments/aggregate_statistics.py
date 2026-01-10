from pathlib import Path
import pandas as pd
import numpy as np
import json
import os


dataset = "german-credit-data"
balanced = 'balanced'

# Load the data
data_path = Path("experiments/output/{dataset}/{balanced}/")
