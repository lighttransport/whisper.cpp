import sys

import torch

import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Need input pytorch model/checkpoint")

checkpoint_filename = sys.argv[1]

checkpoint = torch.load(checkpoint_filename, map_location="cpu")


#target = 'model.encoder.conv1.weight'
target = 'model.encoder.conv2.weight'
weight = checkpoint[target]
print(weight)

print('tensor ', target)
print('min', torch.min(weight))
print('max', torch.max(weight))
hist = torch.histogram(weight, bins=16)
print(hist.hist)

#plt.hist(hist.hist, bins=256)
plt.plot(hist.hist)
plt.show()
