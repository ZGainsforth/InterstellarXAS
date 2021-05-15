import numpy as np
import pandas as pd
import os, sys
from glob2 import glob

FileNames = glob(os.path.join('Orig Data', '*.csv'))

for FileName in FileNames:
    print(FileName)
    # S = np.genfromtxt(FileName, delimiter=',', comments='#') #, skip_header=3)
    S = pd.read_csv(FileName, comment='#', skiprows=1, names=['eV', 'A', 'Counts'])
    S = S.sort_values('eV')
    WriteFileName = os.path.split(FileName)[-1]
    S.to_csv(WriteFileName, columns=['eV', 'Counts'], index=False)
