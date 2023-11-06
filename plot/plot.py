import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##################################
###           GEMVER           ###
##################################

filename = "./results/output_gemver.csv"
df = pd.read_csv(filename, sep=";")

sns.set_theme()
sns.lineplot(data=df, x = "N", y="time [s]", hue="method",errorbar ='sd',
             estimator=np.median)
#plt.xscale('log')

plt.show()
file_path  = "./results/plot_gemver.png"
plt.savefig(file_path)


##################################
###          TRISOLVE          ###
##################################

filename = "./results/output_trisolve.csv"
df = pd.read_csv(filename, sep=";")

sns.set_theme()
sns.lineplot(data=df, x = "N", y="time [s]", hue="method",errorbar ='sd',
             estimator=np.median)
#plt.xscale('log')

plt.show()
file_path  = "./results/plot_trisolve.png"
plt.savefig(file_path)