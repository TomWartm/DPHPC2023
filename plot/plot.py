import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


filename = "./results/output.csv"
df = pd.read_csv(filename, sep=";")

sns.set_theme()
sns.lineplot(
    data=df, x="N", y="time [s]", hue="method", errorbar="sd", estimator=np.median
)
# plt.xscale('log')

plt.show()
file_path = "./results/plot.png"
plt.savefig(file_path)
