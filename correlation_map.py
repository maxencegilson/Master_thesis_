"""
Master Thesis
Academic year 2021-2022

Authors:
    - GILSON Maxence
"""

###########
# Imports #
###########

import matplotlib.pyplot as plt
import pandas as pd
import unsupervised_methods
import seaborn as sns
import used_metrics
from database import DB

# Getting directory
directory = used_metrics.get_directory("corr")
# Creating correlation heatmap
corr_map = DB.corr()
# (16,10) if not binary
plt.figure(figsize=(26, 20))
heatmap = sns.heatmap(corr_map, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(directory + 'With_nul_corr_heatmap_binary.png')
plt.close()

upd_corr_map = corr_map.drop(labels=["CONDUCT Q", "REMEDY 4", "REMEDY 9"], axis=0)
upd_corr_map = upd_corr_map.drop(labels=["CONDUCT Q", "REMEDY 4", "REMEDY 9"], axis=1)

plt.figure(figsize=(26, 20))
heatmap = sns.heatmap(upd_corr_map, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(directory + 'Final_corr_heatmap_binary.png')
plt.close()
