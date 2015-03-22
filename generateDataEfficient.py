import numpy as np
from scipy import io
from random import random
import math
import h5py
from simulatorKurtEfficient import Game, LEVELS
from learnersKurt import activeLearner, reflectiveLearner
import time

channels = 4
images = 100
classes = 2

game = Game(images, channels, classes)
width = game.levelLength
height = LEVELS
start = time.time()

#make data

x, y = game.get()


f = h5py.File("gameData.hdf5", "w")
group = f.create_group("data")
xData = group.create_dataset("x", (images, channels, height, width), dtype='i')
xData[...] = x
yData = group.create_dataset("y", (images,classes), dtype='f')
yData[...] = y

if images<=100:
	np.set_printoptions(threshold=np.nan)
	print(xData[0])
	print(yData[0])

f.close()


f = h5py.File("gameData.hdf5", "r")

print(f["/data/x"])
print(f["/data/y"])
print(time.time()-start, "s")

f.close()

