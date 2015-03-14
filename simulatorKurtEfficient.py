import math
from random import random
from learnersKurt import Learner
from sortedcontainers import SortedListWithKey
import numpy as np

TIME_PER_WAVE = 12
WAVES_PER_LEVEL = 6
TIME_INCREMENT = 0.5
WAIT_TIME_BETWEEN_WAVES = 3
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
	def __init__(self, learner, caracteristics, image, channels):
		self.learner = learner
		self.caracteristics = caracteristics
		self.deadEnemies = {}
		self.beacons = 0
		self.image = image
		self.channels = channels
		self.defTime = 0
		self.defCooldown = 0
		self.beaconTime = 0
		self.lvl = 0
		self.passed = 0

	def act(self, spawn, visible, danger, enemies, totalEnemies, lvl, t):
		if self.lvl != lvl:
			self.lvl = lvl
			self.deadEnemies = {}
			self.passed = 0

		self.defCooldown -= TIME_INCREMENT

		if self.defTime > 0:
			self.defTime -= TIME_INCREMENT
			actionNum = Learner.DEFEND
		elif self.beaconTime > 0:
			self.beaconTime -= TIME_INCREMENT
			actionNum = Learner.BEACON
			if self.beaconTime == 0:
				self.beacons+=1

		enemyNum = totalEnemies - self.passed - len(self.deadEnemies)

		if self.defTime <=0 and self.beaconTime <=0:
			actionNum = self.learner.getAction(self.caracteristics, spawn, visible, danger, self.defCooldown > 0)
			if enemyNum > 0:
				self.doAction(actionNum, enemies)

		#Beacons Act
		if enemyNum>0:
			for i in range(self.beacons):
				target = enemies[math.floor(len(enemies)*random())]
				acc = self.learner.getAcc(self.caracteristics, "BEACON_ACC", target.num)
				if target not in self.deadEnemies and random() < acc:
					self.kill(target)

		for channel in range(self.channels):
			if len(self.image[channel]) <= lvl:
				self.image[channel].append([])
			self.image[channel][lvl][t] = self.drawState(actionNum, channel, enemyNum)


	def doAction(self, actionNum, enemies):
		if actionNum == Learner.SWIPE:
			enemy = self.getClosestEnemy(enemies)
			if random()*100 < self.learner.getAcc(self.caracteristics, "SWIPE_ACC", enemy.num):
				self.kill(enemy)
		elif actionNum == Learner.DEFEND:
			if self.defCooldown <= 0:
				self.defCooldown = Learner.DEFEND_COOLDOWN
				self.defTime = Learner.DEFEND_TIME
			for enemy in self.getDangerEnemies(enemies):
				self.kill(enemy)
		elif actionNum == Learner.BEACON:
			if self.beaconTime <= 0:
				self.beaconTime = Learner.BEACON_TIME

	def enemyPassed(self, enemies):
		for enemy in enemies:
			if enemy not in self.deadEnemies:
				self.passed += 1

	def kill(self, enemy):
		self.deadEnemies[enemy] = True

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
			return action
		else:
			return enemies
	

class Game():
	def __init__(self, number, channels, learners):
		self.levelLength = (TIME_PER_WAVE * WAVES_PER_LEVEL + WAIT_TIME_BETWEEN_WAVES) * (1/TIME_INCREMENT)
		self.channels = channels
		self.learners = learners
		self.number = number
	
	def get(self):
		image = np.empty(shape=(self.number, self.channels, LEVELS, self.levelLength), dtype='i')
		y = np.empty(shape=(self.number,))

		playerInfo = [0]*self.number
		learner = Learner(self.learners)

		for num in range(self.number):
			caracteristics = []
			maxVal = 0
			maxInd = 0
			for i in range(len(self.learners)):
				val = random()
				if val > maxVal:
					maxVal = val
					maxInd = i
				caracteristics.append(val)

			y[num] = maxInd + 1
			playerInfo[num] = Player(learner, caracteristics, image[num], self.channels)

		self.simulate(playerInfo, image)
		return image, y

	def simulate(self, playerInfo, image):

		time = 0
		for lvl in range(LEVELS):
			enemies = SortedListWithKey(None, lambda enemy: enemy.distance)
			self.totalEnemies = 0

			for t in range(math.floor(self.levelLength)):
				time += TIME_INCREMENT
				enemies, passed, spawn, visible, danger = self.nextState(lvl, time, enemies)
		
				for player in playerInfo:
					player.enemyPassed(passed)
					player.act(spawn, visible, danger, enemies, self.totalEnemies, lvl, t)


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

		self.spawnEnemy(enemies, time-level*self.levelLength*TIME_INCREMENT)

		enemiesPassed = []
		while len(enemies)>0 and enemies[0].isDead():
			enemiesPassed.append(enemies[0])
			del enemies[0]

		return enemies, enemiesPassed, spawn, visible, danger


	def spawnEnemy(self, enemies, time):
		if time>=WAIT_TIME_BETWEEN_WAVES and time%1 == 0:
			enemies.add(Enemy(ENEMY_D))
			self.totalEnemies += 1


