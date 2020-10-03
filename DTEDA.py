import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np
from mlxtend.plotting import heatmap

# Load dataset
diabetes = pd.read_csv('diabetes.csv', header=0)
diabetes.columns = ['PREG', 'GLU', 'BP', 'SKIN', 'INSU', 'BMI', 'DPF', 'AGE', 'OUT']
features = ['PREG', 'GLU', 'BP', 'SKIN', 'INSU', 'BMI', 'DPF', 'AGE']
X = diabetes[features].values
y = diabetes['OUT'].T

# EDA
cm = np.corrcoef(diabetes[diabetes.columns].values.T)
hm = heatmap(cm, row_names=diabetes.columns, column_names=diabetes.columns)

scatterplotmatrix(diabetes[diabetes.columns].values, figsize=(10, 8), names=diabetes.columns, alpha=0.4)
plt.show()
