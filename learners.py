from random import random
import math

class Learner:
	WAIT = 0
	SWIPE = 1
	DEFEND = 2
	BEACON = 3

	SWIPE_USE = []
	SWIPE_ACC = []
	DEFEND_USE = []

	def getAction(self, block):
		action = (self.WAIT, 100)
		ran = random()*100
		block = math.floor(block/2)
		if ran < self.SWIPE_USE[block]:
			action = (self.SWIPE, self.SWIPE_ACC[block])
		elif ran < self.SWIPE_USE[block] + self.DEFEND_USE[block]:
			action = (self.DEFEND, 100)

		return action

class ActiveLearner(Learner):
	SWIPE_USE  = [40,60,50,50,40,40,35,35,40,35,35,35,40,35,35,35,40,35,35,35,40,35,35,35,40]
	SWIPE_ACC  = [0,10,15,15,0,15,15,20,0,20,25,25,0,25,25,30,0,35,35,40,0,25,25,30,0]
	DEFEND_USE = [20,10,15,15,10,15,25,30,10,30,30,30,10,30,30,30,10,30,30,30,10,30,30,30,10]

class ReflectiveLearner(Learner):
	SWIPE_USE  = [10,10,15,20,25,20,15,20,25,30,15,15,20,25,30,5,15,20,25,30,5,15,20,25,30,5,15,20,25,30,5]
	SWIPE_ACC  = [25,30,25,20,15,20,25,25,20,15,15,25,25,20,15,5,25,25,20,15,5,25,25,20,15,5,25,25,20,15,5]
	DEFEND_USE = [0,25,25,25,25,0,35,35,35,35,0,50,50,50,50,0,60,60,60,60,0,75,75,75,75,0,70,70,70,70,0]