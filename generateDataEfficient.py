import numpy as np
from random import random
import math
import h5py
from simulatorKurtEfficient import Game, LEVELS, STATE_LENGTH, STORE_IMAGE, STORE_STATE
from learnersKurt import activeLearner, reflectiveLearner
import time

channels = 4
players = 2
images = 10
classes = 2
dataType = STORE_STATE
fileName = "gameData" + str(dataType) + ".hdf5"

game = Game(dataType, images, channels, classes)
start = time.time()

x, y = game.get()
width = game.width
height = game.height


f = h5py.File(fileName, "w")
group = f.create_group("data")
if dataType == STORE_IMAGE:
	xData = group.create_dataset("x", (images, channels, height, width), dtype='i')
	yData = group.create_dataset("y", (images,classes), dtype='f')
else:
	xData = group.create_dataset("x", (images*players, width*height, STATE_LENGTH), dtype='i')
	yData = group.create_dataset("y", (images*players, width*height), dtype='i')
	for p in range(players-1):
		xNext, yNext = game.get()
		x = np.concatenate((x, xNext), axis=0)
		y = np.concatenate((y, yNext), axis=0)
		
		
		
xData[...] = x
yData[...] = y


if images<=100:
	np.set_printoptions(threshold=np.nan)
	ran = math.floor(random()*images)
	#print(xData[ran])
	#print(yData[ran])

print(x.shape, y.shape)

f.close()


print(time.time()-start, "s")

