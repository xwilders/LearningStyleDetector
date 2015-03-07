from random import random
import math
from learners import Learner

BLOCK_TIME = 5
BLOCKS_PER_LEVEL = 4
TIME_INCREMENT = 0.5

LEVELS = 6

class Enemy():
	def __init__(self, distance):
		self.startDistance = distance
		self.distance = distance
		self.dead = False

	def move(self):
		self.distance -= 1
	
	def isClose(self):
		return self.distance <= self.startDistance/3

	def die(self):
		self.dead = True

	def isDead(self):
		if self.dead or self.distance == 0:
			return True
		return False

class Game():
	def __init__(self, levelLength, levels, channels):
		self.levelLength = levelLength
		self.levels = levels
		self.channels = channels

	def drawImage(self, learner):
		result = []
		for c in range(self.channels):
			resultImage = []
			for level in range(self.levels):
				resultImage.append(self.simulate(learner, level, c))
			result.append(resultImage)
		return result

	def simulate(self, learner, level, channel):
		enemies = []
		result = []
		time = 0
		for i in range(math.floor(self.levelLength)):
			enemies, action = self.nextState(learner, level, time, enemies)
			result.append(self.drawState(action, channel, enemies))
		return result

	def nextState(self, learner, level, time, enemies):
		if time == BLOCK_TIME:
			for i in range(10):
				enemies.append(Enemy(self.levelLength - time))

		for enemy in enemies:
			enemy.move()
		
		enemies = [e for e in enemies if not e.isDead()]

		action, accuracy = learner.getAction(level*4 + math.floor(time/BLOCK_TIME))
		self.doAction(action, accuracy, enemies)

		return enemies, action

	def doAction(self, action, accuracy, enemies):
		if len(enemies) > 0:
			if action == Learner.SWIPE:
				if random()*100 < accuracy:
					enemies[0].die()
			elif action == Learner.DEFEND:
				for enemy in enemies:
					if enemy.isClose():
						enemy.die()

	def drawState(self, action, channel, enemies):
		assert (channel>=0 and channel <=1), ('drawState asked to draw unsupported channel '+str(channel))
		if channel == 0:
			return action#len(self.enemies)
		else:
			return len(enemies)



