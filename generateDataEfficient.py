import numpy as np
from random import random
import math
import h5py
from simulatorKurtEfficient import Game, LEVELS, STATE_LENGTH
from learnersKurt import activeLearner, reflectiveLearner
import time

channels = 4
images = 1
classes = 2
dataType = "IMAGE"
fileName = "gameData" + str(dataType) + ".hdf5"

game = Game(dataType, images, channels, classes)
start = time.time()

#make data

x, y = game.get()
width = game.width
height = game.height


f = h5py.File(fileName, "w")
group = f.create_group("data")
if dataType == "IMAGE":
	xData = group.create_dataset("x", (images, channels, height, width), dtype='i')
	yData = group.create_dataset("y", (images,classes), dtype='f')
else:
	xData = group.create_dataset("x", (images, width*height, STATE_LENGTH), dtype='i')
	yData = group.create_dataset("y", (images, width*height), dtype='i')

xData[...] = x
yData[...] = y


if images<=100:
	np.set_printoptions(threshold=np.nan)
	ran = math.floor(random()*images)
	print(xData[ran])
	print(yData[ran])

print(x.shape, y.shape)

f.close()


print(time.time()-start, "s")

