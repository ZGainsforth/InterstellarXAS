import numpy as np
import pandas as pd
import os, sys
from glob2 import glob

FileNames = glob(os.path.join('Orig Data', '*.csv'))

for FileName in FileNames:
    print(FileName)
    S = pd.read_csv(FileName)
    S = S.sort_values('eV')
    print(S.head())
    WriteFileName = os.path.split(FileName)[-1]
    S.to_csv(WriteFileName, columns=['eV', 'COUNTS'], index=False)
