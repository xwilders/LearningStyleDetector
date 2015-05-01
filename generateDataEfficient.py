import numpy as np
from random import random
import math
import h5py
from simulatorKurtEfficient import Game, LEVELS, STATE_LENGTH, STORE_IMAGE, STORE_STATE
from learnersKurt import activeLearner, reflectiveLearner
import time

channels = 6
players = 1
images = 100
classes = 2
dataType = STORE_IMAGE #STORE_STATE
fileName = "gameData" + str(dataType) + ".hdf5"

game = Game(dataType, images, channels, classes)
start = time.time()

width = game.width
height = game.height


f = h5py.File(fileName, "w")
group = f.create_group("data")
if dataType == STORE_IMAGE:
	xData = group.create_dataset("x", (images, channels, height, width), dtype='i')
	yData = group.create_dataset("y", (images,classes), dtype='f')
	xFull, yFull, player = game.get()
else:
	xData = group.create_dataset("x", (images*players, width*height, STATE_LENGTH), dtype='i')
	yData = group.create_dataset("y", (images*players, width*height), dtype='i')
	playerData = group.create_dataset("players", (players, ), dtype='f')

	xFull = np.empty(shape=(images*players, width*height, STATE_LENGTH), dtype='i')
	yFull = np.empty(shape=(images*players, width*height), dtype='i')
	playerFull = np.empty(shape=(players, ), dtype='f')

	for p in range(players):
		x, y, player = game.get()
		for i in range(images):
			xFull[p*images + i] = x[i]
			yFull[p*images + i] = y[i]
		playerFull[p] = player
		if p%100==0 and p>0:
			print(p)

	playerData[...] = playerFull


xData[...] = xFull
yData[...] = yFull


if images<=100:
	np.set_printoptions(threshold=np.nan)
	ran = math.floor(random()*images)
	#print(xData[ran])
	#print(yData[ran])
	#print(playerData)

print(xFull.shape, yFull.shape)

f.close()


print(time.time()-start, "s")

