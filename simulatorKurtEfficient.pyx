import math
from random import random
from learnersKurt import Learner
import numpy as np
from libc.stdlib cimport rand, RAND_MAX
import time

cdef int SEC_PER_WAVE = 12
cdef int WAVES_PER_LEVEL = 6
cdef int INCR_PER_SEC = 2
cdef float TIME_INCR = 1.0/INCR_PER_SEC
cdef int START_WAVE_SEC = 0
cdef float WAVE_END_SEC = 26 * TIME_INCR

STATE_LENGTH = 10
LEVELS = 6

STORE_IMAGE = 0
STORE_STATE = 1

cdef int ENEMIES_PER_WAVE = 24

cdef int ENEMY_D = 0
cdef int ENEMY_S = 1
cdef int ENEMY_Z = 2
cdef int ENEMY_R = 3

cdef class Enemy:
	cdef public int num
	cdef float speed
	cdef public bint boss
	cdef int distanceToVis
	cdef int distanceFromVis
	cdef public float distance

	def __init__(self, int enemyNum, speed = 1, boss = False):
		self.num = enemyNum
		self.speed = speed
		self.boss = boss
		if enemyNum == ENEMY_D:
			self.distanceToVis = rand()%5 + 1
			self.distanceFromVis = rand()%4 + 7
		elif enemyNum == ENEMY_S:
			self.distanceToVis = rand()%7 + 3
			self.distanceFromVis = rand()%6 + 13
		elif enemyNum == ENEMY_Z:
			self.distanceToVis = rand()%6 + 1
			self.distanceFromVis = rand()%5 + 8
		elif enemyNum == ENEMY_R:
			self.distanceToVis = rand()%8 + 1
			self.distanceFromVis = rand()%12 + 7
		self.distance = self.distanceToVis + self.distanceFromVis

	cpdef move(self):
		self.distance -= self.speed
	
	cpdef bint isVisible(self):
		return self.distance <= self.distanceFromVis

	cpdef bint isInDanger(self):
		return self.distance <= 3

	cpdef bint isDead(self):
		return self.distance <= 0

class Player():

	def __init__(self, storageType, learner, x, y, int channels, deadEnemies):
		self.learner = learner
		self.deadEnemies = deadEnemies
		self.x = x
		self.y = y
		self.storageIndex = 0
		self.channels = channels
		self.defTime = 0
		self.defCooldown = 0
		self.beaconTime = 0
		self.healthLost = 0
		self.beacons = 0
		self.storageType = storageType
		self.lastAction = 0

	def act(self, int spawn, int visible, int danger, enemies, int totalEnemies, int lvl, int timeInWaveIncr, int timeInLvl, visibleType, dangerType):
		if timeInWaveIncr == 0:
			self.passed = 0
			self.killedThisWave = 0
		if timeInLvl == 0:
			self.beacons = 0

		cdef int enemyNum = totalEnemies - self.passed - self.killedThisWave
		cdef int actionNum = Learner.NONE
		cdef int i

		if self.killedThisWave+self.passed < ENEMIES_PER_WAVE: # or timeInWaveIncr < ENEMIES_PER_WAVE + START_WAVE_SEC*INCR_PER_SEC:

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

			if (self.beaconTime > 1 and danger > 0) or danger > 1:
				self.beaconTime = 0 #Cancel beacon

			if self.defTime <=0 and self.beaconTime <=0:
				actionNum = self.learner.getAction(spawn, visible, danger, self.defCooldown > 0)

			self.doAction(actionNum, enemies, enemyNum)

			#Beacons Act
			if timeInWaveIncr % 2 == 0:
				for i in range(self.beacons):
					enemyNum = totalEnemies - self.passed - self.killedThisWave
					if enemyNum>0:
						target = self.getFurthestEnemy(enemies) #enemies[rand()%enemyNum] #Improve this???
						if target is not None:
							acc = self.learner.getAcc("BEACON_ACC", target.num)
							if self.notDead(target) and rand()%100 < acc:
								self.kill(target)

		if self.storageType == STORE_IMAGE:
			for channel in range(self.channels):
				if len(self.x[channel]) <= lvl:
					self.x[channel].append([])
				self.x[channel][lvl][timeInLvl] = self.drawState(actionNum+1, channel, enemyNum)
		else:#if self.killedThisWave < ENEMIES_PER_WAVE:
			#Only store valid game time
			self.x[self.storageIndex] = self.getState(visibleType, dangerType, self.getClosestEnemy(enemies), timeInWaveIncr, timeInLvl)
			self.y[self.storageIndex] = actionNum + 1 #Crashes Lua if 0
			self.lastAction = actionNum
			self.storageIndex += 1


	def doAction(self, int actionNum, enemies, int enemyNum):
		cdef Enemy enemy
		if actionNum == Learner.SWIPE and enemyNum>0:
			enemy = self.getClosestEnemy(enemies)
			if enemy is not None and rand()%100 < self.learner.getAcc("SWIPE_ACC", enemy.num):
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

	def notDead(self, Enemy enemy):
		return (self, enemy) not in self.deadEnemies or self.deadEnemies[(self, enemy)]>0

	def enemyPassed(self, enemies):
		cdef Enemy enemy
		for enemy in enemies:
			if self.notDead(enemy):
				self.passed += 1
				self.healthLost += 1
				if enemy.boss:
					self.healthLost += 1

	def kill(self, Enemy enemy):
		cdef int life = 0
		if enemy.boss:
			life = 2
			if (self, enemy) in self.deadEnemies:
				life = self.deadEnemies[(self, enemy)] - 1
		self.deadEnemies[(self, enemy)] = life
		
		if life == 0:
			self.killedThisWave += 1

	def getClosestEnemy(self, enemies):
		cdef Enemy enemy
		for enemy in enemies:
			if not enemy.isVisible():
				return None
			if self.notDead(enemy):
				return enemy

	def getFurthestEnemy(self, enemies):
		cdef int i
		cdef int length = len(enemies)
		cdef Enemy enemy
		for i in range(length):
			enemy = enemies[length - i - 1]
			if self.notDead(enemy) and enemy.isVisible():
				return enemy

	def getDangerEnemies(self, enemies):
		cdef Enemy enemy
		for enemy in enemies:
			if not enemy.isInDanger():
					break
			if self.notDead(enemy):
				yield enemy

	def drawState(self, int action, int channel, int enemyNum):
		if channel == 0:
			return action
		elif channel == 1:
			return enemyNum
		elif channel == 2:
			return self.beacons
		elif channel == 3:
			return self.healthLost

	def getState(self, vT, dT, closestEnemy, timeInWaveIncr, timeInLvl):
		num = 0
		distance = 0
		if closestEnemy is not None:
			num = closestEnemy.num
			distance = closestEnemy.distance
		return np.array([vT[0]+ vT[1]+ vT[2]+ vT[3], dT[0]+dT[1]+dT[2]+dT[3], num, distance, self.lastAction+1, self.defCooldown, self.beacons, timeInWaveIncr, timeInLvl, self.healthLost])


		'''
		State: (Health, #Beacons, Action Taken, #Enemies)

		All data:
		Number of enemies, and of what type: 4
		Action at every timestep: 4
		Number of beacons: 1
		Health: 1
		'''
	

class Game():
	def __init__(self, storageType, int number, int channels, int classes):
		self.waveLength = (START_WAVE_SEC + SEC_PER_WAVE + WAVE_END_SEC) * INCR_PER_SEC
		self.levelLength = self.waveLength*WAVES_PER_LEVEL
		self.width = self.levelLength
		self.height = LEVELS
		self.channels = channels
		self.number = number
		self.deadEnemies = {}
		self.classes = classes
		self.storageType = storageType
	
	def get(self):
		if self.storageType == STORE_IMAGE:
			x = np.empty(shape=(self.number, self.channels, LEVELS, self.levelLength), dtype='i')
			y = np.empty(shape=(self.number,self.classes), dtype='f')
		else:
			x = np.empty(shape=(self.number, self.width * self.height, STATE_LENGTH), dtype='i')
			y = np.empty(shape=(self.number, self.width * self.height), dtype='i')


		playerInfo = [0]*self.number

		cdef int num
		cdef float active

		active = (rand()%1000)/1000.0
		for num in range(self.number):
			if self.storageType == STORE_IMAGE:
				active = (rand()%1000)/1000.0 #Different people
			arr = [active, 1 - active]
			learner = Learner(arr)
			if self.storageType == STORE_IMAGE:
				y[num] = arr

			playerInfo[num] = Player(self.storageType, learner, x[num], y[num], self.channels, self.deadEnemies)

		self.simulate(playerInfo, x)
		return x, y

	def simulate(self, playerInfo, image):

		for lvl in range(LEVELS):
			enemies = []
			if self.storageType == STORE_IMAGE:
				print("Starting level", lvl)
			self.totalEnemies = 0
			timeInLvl = -1

			for wave in range(WAVES_PER_LEVEL):
				startW = time.time()
				for t in range(math.floor(self.waveLength)):
					timeInLvl += 1
					enemies, passed, spawn, visible, danger, visibleType, dangerType = self.nextState(lvl, wave, t, enemies)
			
					for player in playerInfo:
						player.enemyPassed(passed)
						player.act(spawn, visible, danger, enemies, self.totalEnemies, lvl, t, timeInLvl, visibleType, dangerType)

				self.totalEnemies = 0
				self.deadEnemies.clear()
				#print("wave", time.time() - startW)


	def nextState(self, int level, int wave, int time, enemies):
		cdef int spawn = 0
		cdef int visible = 0
		cdef int danger = 0
		cdef Enemy enemy

		visibleType = [0,0,0,0]
		dangerType = [0,0,0,0]

		enemiesPassed = []

		if time>START_WAVE_SEC*INCR_PER_SEC:
			self.spawnEnemy(enemies, time, level, wave)

			for enemy in enemies:
				enemy.move()
				if enemy.isInDanger():
					danger += 1
					dangerType[enemy.num] += 1
				elif enemy.isVisible():
					visible += 1
					visibleType[enemy.num] += 1
				else:
					spawn += 1

			enemies.sort(key = lambda e:e.distance)

			while len(enemies)>0 and enemies[0].isDead():
				enemiesPassed.append(enemies[0])
				del enemies[0]

		return enemies, enemiesPassed, spawn, visible, danger, visibleType, dangerType


	def spawnEnemy(self, enemies, int timeIncr, int lvl, int wave):
		cdef Enemy enemy
		if self.totalEnemies < ENEMIES_PER_WAVE:
			enemy = self.getNextEnemy(lvl, wave, self.totalEnemies)
			enemies.append(enemy)
			self.totalEnemies += 1

	def getNextEnemy(self, int lvl, int wave, int num):
		cdef int enemy
		if lvl == 0:
			enemy = ENEMY_D
		elif lvl == 1:
			enemy = ENEMY_S
		elif lvl == 2:
			enemy = ENEMY_Z
		elif lvl == 3:
			enemy = ENEMY_R
		elif lvl == 4:
			if math.floor(num/4)%2 == 0:
				enemy = ENEMY_D
			else:
				enemy = ENEMY_S
		elif lvl == 5:
			if math.floor(num/3)%4 == 0:
				enemy = ENEMY_D
			elif math.floor(num/3)%4 == 1:
				enemy = ENEMY_S
			elif math.floor(num/3)%4 == 2:
				enemy = ENEMY_Z
			elif math.floor(num/3)%4 == 3:
				enemy = ENEMY_R

		cdef float speed = 1
		cdef bint boss = False
		if wave == 5:
			speed = 1.5
			if num%6==0:
				speed = 1
				boss = True
		elif wave == 4 or (wave > 0 and num%4 == 0) or (wave > 1 and num%3 == 0) or (wave > 2 and num%2 == 0):
			speed = 1.5
		
		return Enemy(enemy, speed, boss)












