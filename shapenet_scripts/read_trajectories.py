import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
args = parser.parse_args()

with open(args.file, 'rb') as fp:
    trajectories = pickle.load(fp)

assert len(trajectories) > 0
k = 0
for i in trajectories:
    if(i['rewards'][len(i['rewards'])-1] != 0):   
        k += 1

print(k / len(trajectories))
