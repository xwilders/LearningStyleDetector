from random import random
import math
from learnersKurt import Learner
from sortedcontainers import SortedListWithKey

TIME_PER_WAVE = 12
WAVES_PER_LEVEL = 6
TIME_INCREMENT = 0.5
LEVELS = 6

ENEMY_D = (1, 1)

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

	def die(self):
		self.distance = -1
	

class Game():
	def __init__(self, channels):
		self.levelLength = TIME_PER_WAVE * WAVES_PER_LEVEL * (1/TIME_INCREMENT)
		self.channels = channels

	def drawImage(self, learner):
		wholeImage = []
		for c in range(self.channels):
			levelImage = []
			for level in range(LEVELS):
				levelImage.append(self.simulate(learner, level, c))
			wholeImage.append(levelImage)
		return wholeImage

	def simulate(self, learner, level, channel):
		enemies = SortedListWithKey(None, lambda enemy: enemy.distance)
		levelImage = []
		time = 0
		for i in range(math.floor(self.levelLength)):
			time += TIME_INCREMENT
			enemies, action = self.nextState(learner, level, time, enemies)
			levelImage.append(self.drawState(action, channel, enemies))
		return levelImage

	def nextState(self, learner, level, time, enemies):
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

		action, accuracy = learner.getAction(spawn, visible, danger, enemies[0].num)
		self.doAction(action, accuracy, enemies)

		while len(enemies)>0 and enemies[0].isDead():
			del enemies[0]

		return enemies, action

	def doAction(self, action, accuracy, enemies):
		if len(enemies) > 0:
			if action == Learner.SWIPE:
				if random()*100 < accuracy:
					enemies[0].die()
			elif action == Learner.DEFEND:
				for enemy in enemies:
					if not enemy.isInDanger():
						break
					if random()*100 < accuracy[enemy.num]:
						enemy.die()


	def spawnEnemy(self, time):
		return Enemy(ENEMY_D)


	def drawState(self, action, channel, enemies):
		assert (channel>=0 and channel <=1), ('drawState asked to draw unsupported channel '+str(channel))
		if channel == 0:
			return action#len(self.enemies)
		else:
			return len(enemies)



