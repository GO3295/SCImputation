import pandas as pd
#import numpy as np
#import os
#import time
#from sklearn.utils.extmath import randomized_svd
#import random
#import sys
from afMF.runafMF import afMF

dat = pd.read_csv("/restricted/projectnb/casa/jh50/jinghan/others/SCI/rawcount_txt/GSE75748_sc_cell_type_genebycell.txt", sep="\t", index_col=0)
imputed_dat = afMF(dat)
print(imputed_dat.iloc[:, : 5].head(5))
