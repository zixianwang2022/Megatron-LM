import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('name', metavar='N', type=str, help='csv file')

args = parser.parse_args()

print(args.name)

df = pd.read_csv(args.name)#, index_col=0)
#print(df.columns)

loss = df['lm loss vs samples'].to_numpy()
#loss = df['num-zeros vs samples'].to_numpy()
#loss = df['grad-norm vs samples'].to_numpy()

n = 1000
largest_index = int(loss.shape[0] / n) * n
#largest_index = 1000
loss = loss[:largest_index]

loss[np.isnan(loss)] = 0

print(loss.shape)

avg = np.mean(loss.reshape(-1, n), axis=1)
std = np.std(loss.reshape(-1, n), axis=1)

print(avg.shape)

for i in range(avg.shape[0]):
    print(avg[i])#,std[i])
