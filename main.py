import numpy as np
import pandas as pd
import re
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import os
import random
# import matplotlib.pyplot as plt
# import seaborn as sns

#GPU check
is_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_cuda else "cpu")
print("Device:", device)