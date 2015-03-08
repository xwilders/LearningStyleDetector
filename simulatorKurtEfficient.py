from random import random
import math
from learnersKurt import Learner
from sortedcontainers import SortedListWithKey
import numpy as np

TIME_PER_WAVE = 12
WAVES_PER_LEVEL = 6
TIME_INCREMENT = 0.5
#WAIT_TIME_BETWEEN_WAVES = 3
LEVELS = 6

ENEMY_D = (0, 1)

class Enemy():
	def __init__(self, enemy):
		self.num = enemy[0]
		self.speed = enemy[1]
		self.distanceToVis = math.floor(random()*5) + 1
		self.distanceFromVis = math.floor(random()*4) + 7
		self.distance = self.distanceToVis + self.distanceFromVis

	def move(self):
		self.distance -= self.speed
	
	def isVisible(self):
		return self.distance <= self.distanceFromVis

	def isInDanger(self):
		return self.distance <= 2

	def isDead(self):
		return self.distance <= 0

class Player():
	def __init__(self, learner, image, channels):
		self.learner = learner
		self.deadEnemies = []
		self.image = image
		self.channels = channels

	def act(self, spawn, visible, danger, enemies, lvl, t):
		actionNum, actionAcc = self.learner.getAction(spawn, visible, danger)
		if len(enemies) > 0:
			self.doAction(actionNum, actionAcc, enemies)
		for channel in range(self.channels):
			if len(self.image[channel]) <= lvl:
				self.image[channel].append([])
			self.image[channel][lvl][t] = self.drawState(actionNum, channel, enemies)


	def doAction(self, actionNum, accVector, enemies):
		if actionNum == Learner.SWIPE:
			enemy = self.getClosestEnemy(enemies)
			if random()*100 < accVector[enemy.num]:
				self.deadEnemies.append(enemy)
		elif actionNum == Learner.DEFEND:
			for enemy in self.getDangerEnemies(enemies):
				if random()*100 < accVector[enemy.num]:
					self.deadEnemies.append(enemy)

	def getClosestEnemy(self, enemies):
		for enemy in enemies:
			if enemy not in self.deadEnemies:
				return enemy

	def getDangerEnemies(self, enemies):
		for enemy in enemies:
			if not enemy.isInDanger():
					break
			if enemy not in self.deadEnemies:
				yield enemy

	def drawState(self, action, channel, enemies):
		assert (channel>=0 and channel <=1), ('drawState asked to draw unsupported channel '+str(channel))
		if channel == 0:
			return action#len(self.enemies)
		else:
			return len(enemies)
	

class Game():
	def __init__(self, number, channels, learners):
		self.levelLength = TIME_PER_WAVE * WAVES_PER_LEVEL * (1/TIME_INCREMENT)
		self.channels = channels
		self.learners = learners
		self.number = number
	
	def get(self):
		image = np.empty(shape=(self.number, self.channels, LEVELS, self.levelLength), dtype='i')
		y = np.empty(shape=(self.number,))

		playerInfo = [0]*self.number

		for num in range(self.number):
			classL = math.floor(random()*2)
			y[num] = classL + 1
			playerInfo[num] = Player(self.learners[classL], image[num], self.channels)

		self.simulate(playerInfo, image)
		return image, y

	def simulate(self, playerInfo, image):

		time = 0
		for lvl in range(LEVELS):
			enemies = SortedListWithKey(None, lambda enemy: enemy.distance)

			for t in range(math.floor(self.levelLength)):
				time += TIME_INCREMENT
				enemies, spawn, visible, danger = self.nextState(lvl, time, enemies)
		
				for player in playerInfo:
					player.act(spawn, visible, danger, enemies, lvl, t)


	def nextState(self, level, time, enemies):
		spawn = 1
		visible = 0
		danger = 0

		for enemy in enemies:
			enemy.move()
			if enemy.isInDanger():
				danger += 1
			elif enemy.isVisible():
				visible += 1
			else:
				spawn += 1

		enemies.add(self.spawnEnemy(time))

		while len(enemies)>0 and enemies[0].isDead():
			del enemies[0]

		return enemies, spawn, visible, danger


	def spawnEnemy(self, time):
		return Enemy(ENEMY_D)



