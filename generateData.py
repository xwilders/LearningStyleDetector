import numpy as np
from scipy import io
from random import random
import math
import h5py
from simulator import Game, BLOCK_TIME, TIME_INCREMENT, BLOCKS_PER_LEVEL, LEVELS
from learners import ActiveLearner, ReflectiveLearner

width = BLOCK_TIME * BLOCKS_PER_LEVEL * (1/TIME_INCREMENT)
height = LEVELS
channels = 1
images = 10000

game = Game(width, height, channels)
classes = [ActiveLearner(), ReflectiveLearner()]

#make data

x = np.empty(shape=(images, channels, height, width), dtype='i')
y = np.empty(shape=(images,))

def drawByClass(imageNum, arr):
	assert (arr.shape==(channels, height, width)), ('drawByClass expects input of shape '+str(width* height))
	
	label = math.floor(random()*2)
	x[image] = game.drawImage(classes[label])

	return label+1

for image in range(images):
	y[image] = drawByClass(image, x[image])


f = h5py.File("gameData.hdf5", "w")
group = f.create_group("data")
xData = group.create_dataset("x", (images, channels, height, width), dtype='i')
xData[...] = x
yData = group.create_dataset("y", (images,), dtype='i')
yData[...] = y

print(xData[0:3])
print(yData[0:3])

f.close()


f = h5py.File("gameData.hdf5", "r")

print(f["/data/x"])
print(f["/data/y"])

f.close()

