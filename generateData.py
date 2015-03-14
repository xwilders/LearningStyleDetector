import numpy as np
from scipy import io
from random import random
import math
import h5py
from simulatorKurt import Game, LEVELS
from learnersKurt import ActiveLearner, ReflectiveLearner
import time

channels = 1
images = 100

game = Game(channels)
width = game.levelLength
height = LEVELS
classes = [ActiveLearner(), ReflectiveLearner()]
start = time.time()

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
	if image>0 and image%(images/10) == 0 and images >= 1000:
		print(image, "images generated")


f = h5py.File("gameData.hdf5", "w")
group = f.create_group("data")
xData = group.create_dataset("x", (images, channels, height, width), dtype='i')
xData[...] = x
yData = group.create_dataset("y", (images,), dtype='i')
yData[...] = y

#print(xData[0:3])
#print(yData[0:3])

f.close()


f = h5py.File("gameData.hdf5", "r")

print(f["/data/x"])
print(f["/data/y"])
print(time.time()-start, "s")

f.close()

