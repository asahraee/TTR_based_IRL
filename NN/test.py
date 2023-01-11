import numpy
import pandas as pd

l_path = '/root/Desktop/data_and_log/data_2/csv/image_labels.csv'

ls = pd.read_csv(l_path, sep=',', header=None, index_col =0)

print(ls.iloc[:,0])

test = ls.iloc[1000,0].split('_')[0]
test2 = test[0]+'_'+test[1:]
print(test2)
