import math
from random import random
from learnersKurt import Learner
from sortedcontainers import SortedListWithKey
import numpy as np

cdef extern from "stdlib.h":
		int rand "rand"()

SEC_PER_WAVE = 12
WAVES_PER_LEVEL = 6
INCR_PER_SEC = 2
TIME_INCR = 1/INCR_PER_SEC
START_WAVE_SEC = 3
WAVE_END_SEC = 7.5
LEVELS = 6

ENEMIES_PER_WAVE = 24

ENEMY_D = 0
ENEMY_S = 1
ENEMY_Z = 2
ENEMY_R = 3

class Enemy():
	def __init__(self, enemyNum, speed = 1, boss = False):
		self.num = enemyNum
		self.speed = speed
		self.boss = boss
		self.distanceToVis = rand()%5 + 1
		self.distanceFromVis = rand()%4 + 7
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
	def __init__(self, learner, image, int channels, deadEnemies):
		self.learner = learner
		self.deadEnemies = deadEnemies
		self.beacons = 0
		self.image = image
		self.channels = channels
		self.defTime = 0
		self.defCooldown = 0
		self.beaconTime = 0
		self.lvl = 0
		self.healthLost = 0

	def act(self, int spawn, int visible, int danger, enemies, int totalEnemies, int lvl, int timeInWaveIncr, int timeInLvl):
		if not self.deadEnemies:
			self.passed = 0
			self.killedThisWave = 0

		cdef int enemyNum = totalEnemies - self.passed - self.killedThisWave
		cdef int i

		if self.killedThisWave < ENEMIES_PER_WAVE: # or timeInWaveIncr < ENEMIES_PER_WAVE + START_WAVE_SEC*INCR_PER_SEC:

			if self.lvl != lvl:
				self.lvl = lvl
				self.passed = 0

			self.defCooldown -= TIME_INCR
			actionNum = Learner.WAIT

			if self.defTime > 0:
				self.defTime -= TIME_INCR
				actionNum = Learner.DEFEND
			elif self.beaconTime > 0:
				self.beaconTime -= TIME_INCR
				actionNum = Learner.BEACON
				if self.beaconTime == 0:
					self.beacons+=1

			if self.defTime <=0 and self.beaconTime <=0:
				actionNum = self.learner.getAction(spawn, visible, danger, self.defCooldown > 0)

			self.doAction(actionNum, enemies, enemyNum)

			#Beacons Act
			for i in range(self.beacons):
				if enemyNum>0:
					target = enemies[rand()%len(enemies)]
					acc = self.learner.getAcc("BEACON_ACC", target.num)
					if self.notDead(target) and rand()%100 < acc:
						self.kill(target)
		else:
			actionNum = Learner.NONE

		for channel in range(self.channels):
			if len(self.image[channel]) <= lvl:
				self.image[channel].append([])
			self.image[channel][lvl][timeInLvl] = self.drawState(actionNum+1, channel, enemyNum)


	def doAction(self, int actionNum, enemies, int enemyNum):
		if actionNum == Learner.SWIPE and enemyNum>0:
			enemy = self.getClosestEnemy(enemies)
			if rand()%100 < self.learner.getAcc("SWIPE_ACC", enemy.num):
				self.kill(enemy)
		elif actionNum == Learner.DEFEND:
			if self.defCooldown <= 0:
				self.defCooldown = Learner.DEFEND_COOLDOWN
				self.defTime = Learner.DEFEND_TIME
			if enemyNum > 0:
				for enemy in self.getDangerEnemies(enemies):
					self.kill(enemy)
		elif actionNum == Learner.BEACON:
			if self.beaconTime <= 0:
				self.beaconTime = Learner.BEACON_TIME

	def notDead(self, enemy):
		return (self, enemy) not in self.deadEnemies or self.deadEnemies[(self, enemy)]>0

	def enemyPassed(self, enemies):
		for enemy in enemies:
			if self.notDead(enemy):
				self.passed += 1
				self.healthLost += 1

	def kill(self, enemy):
		life = 0
		if enemy.boss:
			life = 2
			if (self, enemy) in self.deadEnemies:
				life = self.deadEnemies[(self, enemy)] - 1
		self.deadEnemies[(self, enemy)] = life
		
		if life == 0:
			self.killedThisWave += 1

	def getClosestEnemy(self, enemies):
		for enemy in enemies:
			if self.notDead(enemy):
				return enemy

	def getDangerEnemies(self, enemies):
		for enemy in enemies:
			if not enemy.isInDanger():
					break
			if self.notDead(enemy):
				yield enemy

	def drawState(self, int action, int channel, enemies):
		assert (channel>=0 and channel <=3), ('drawState asked to draw unsupported channel '+str(channel))
		if channel == 0:
			return action
		elif channel == 1:
			return enemies
		elif channel == 2:
			return self.beacons
		elif channel == 3:
			return self.healthLost


		'''
		All data:
		Number of enemies, and of what type: 4
		Action at every timestep: 4
		Number of beacons: 1
		Health: 1
		'''
	

class Game():
	def __init__(self, number, channels, classes):
		self.waveLength = (SEC_PER_WAVE + WAVE_END_SEC) * INCR_PER_SEC
		self.levelLength = self.waveLength*WAVES_PER_LEVEL + START_WAVE_SEC*INCR_PER_SEC
		self.channels = channels
		self.number = number
		self.deadEnemies = {}
		self.classes = classes
	
	def get(self):
		image = np.empty(shape=(self.number, self.channels, LEVELS, self.levelLength), dtype='i')
		y = np.empty(shape=(self.number,self.classes), dtype='f')

		playerInfo = [0]*self.number

		for num in range(self.number):
			active = random()
			reflective = 1 - active
			y[num] = [active, reflective]
			learner = Learner(y[num])

			playerInfo[num] = Player(learner, image[num], self.channels, self.deadEnemies)

		self.simulate(playerInfo, image)
		return image, y

	def simulate(self, playerInfo, image):

		for lvl in range(LEVELS):
			enemies = SortedListWithKey(None, lambda enemy: enemy.distance)
			print("Starting level", lvl)
			self.totalEnemies = 0
			timeInLvl = -1

			for t in range(START_WAVE_SEC * INCR_PER_SEC):
				timeInLvl += 1
				for player in playerInfo:
					player.act(0, 0, 0, enemies, 0, lvl, t, timeInLvl)

			for wave in range(WAVES_PER_LEVEL):
				for t in range(math.floor(self.waveLength)):
					timeInLvl += 1
					enemies, passed, spawn, visible, danger = self.nextState(lvl, wave, t, enemies)
			
					for player in playerInfo:
						player.enemyPassed(passed)
						player.act(spawn, visible, danger, enemies, self.totalEnemies, lvl, t, timeInLvl)

				self.totalEnemies = 0
				self.deadEnemies.clear()


	def nextState(self, level, wave, time, enemies):
		spawn = 0
		visible = 0
		danger = 0

		self.spawnEnemy(enemies, time, level, wave)

		for enemy in enemies:
			enemy.move()
			if enemy.isInDanger():
				danger += 1
			elif enemy.isVisible():
				visible += 1
			else:
				spawn += 1

		enemiesPassed = []
		while len(enemies)>0 and enemies[0].isDead():
			enemiesPassed.append(enemies[0])
			del enemies[0]

		return enemies, enemiesPassed, spawn, visible, danger


	def spawnEnemy(self, enemies, timeIncr, lvl, wave):
		if self.totalEnemies < ENEMIES_PER_WAVE:
			enemy = self.getNextEnemy(lvl, wave, self.totalEnemies)
			enemies.add(enemy)
			self.totalEnemies += 1

	def getNextEnemy(self, lvl, wave, num):
		if lvl == 0:
			enemy = ENEMY_D
		elif lvl == 1:
			enemy = ENEMY_S
		elif lvl == 2:
			enemy = ENEMY_Z
		elif lvl == 3:
			enemy = ENEMY_R
		elif lvl == 4:
			enemy = rand()%4
		elif lvl == 5:
			enemy = rand()%4

		speed = 1
		boss = False
		if wave == 5:
			speed = 2
			if num%6==0:
				boss = True
		elif wave == 4 or (wave > 0 and num%4 == 0) or (wave > 1 and num%3 == 0) or (wave > 2 and num%2 == 0):
			speed = 2
		
		return Enemy(enemy, speed, boss)












