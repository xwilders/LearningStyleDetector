from random import random
import math

class Learner:
	
	WAIT = 0
	SWIPE = 1
	BEACON = 2
	DEFEND = 3

	def getAction(self, spawn, visible, danger, enemyType):
		action = (self.WAIT, 100)
		ran = random()*100
		situation = self.CLEAR
		if danger>=2:
			situation = self.DANGER_2
		elif danger==1:
			situation = self.DANGER_1
		elif visible>=5:
			situation = self.VISIBLE_3
		elif visible >=3:
			situation = self.VISIBLE_2
		elif visible >=1:
			situation = self.VISIBLE_1

		if ran < situation[self.SWIPE]:
			action = (self.SWIPE, self.SWIPE_ACC[enemyType])
		elif ran < situation[self.BEACON] + situation[self.SWIPE]:
			action = (self.BEACON, 100)
		elif ran < situation[self.DEFEND] + situation[self.BEACON] + situation[self.SWIPE]:
			action = (self.DEFEND, self.DEFEND_ACC)
		return action

class ReflectiveLearner(Learner):
	CLEAR = [10, 0,90,0]
	SPAWN = [5, 0,95,0]
	VISIBLE_1 = [0, 20,80,0]
	VISIBLE_2 = [0, 25,75,0]
	VISIBLE_3 = [0, 50,50,0]
	DANGER_1 = [0, 40,40,20]
	DANGER_2 = [0, 10,5,85]

	SWIPE_ACC = [30,10,20,25]
	BEACON_ACC = [15,35,25,30]
	DEFEND_ACC = [90,70,60,80]


class ActiveLearner(Learner):
	CLEAR = [20, 20,50,10]
	SPAWN = [10, 25,60,5]
	VISIBLE_1 = [10, 80,10,0]
	VISIBLE_2 = [5, 85,10,0]
	VISIBLE_3 = [10, 85,5,0]
	DANGER_1 = [0, 80,0,20]
	DANGER_2 = [0, 30,0,70]

	SWIPE_ACC = [40,20,30,35]
	BEACON_ACC = [10,30,20,25]
	DEFEND_ACC = [85,60,50,80]