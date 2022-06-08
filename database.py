"""
Master Thesis
Academic year 2021-2022

Authors:
    - GILSON Maxence
"""

###########
# Imports #
###########
import numpy as np
import pandas as pd

"""
Extracts the Excel file
"""

file_path = '/Users/maxence/Desktop/PycharmProjects/TFE/Data/Data_final.xlsx'
sheet_name = 1
header = 0
index_col = None
DB_Cat = pd.read_excel(file_path, sheet_name=1, header=0, index_col=None)
DB_Cat = DB_Cat.drop(labels=["Number cluster"], axis=1)
DB_Cat = DB_Cat.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
DB_Cat = DB_Cat.drop(labels=["CONDUCT Q", "REMEDY 4", "REMEDY 9"], axis=1)

# Binary/discrete database
DB = pd.get_dummies(DB_Cat, drop_first=False)
# Numpy database
DB_np = DB.to_numpy()
