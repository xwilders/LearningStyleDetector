from random import gauss
import math

cdef extern from "stdlib.h":
		int rand "rand"()

class Learner:
	
	NONE = -1
	WAIT = 0
	SWIPE = 1
	BEACON = 2
	DEFEND = 3

	DEFEND_TIME = 1
	DEFEND_COOLDOWN = 2
	BEACON_TIME = 3

	def __init__(self, learner):
		cdef int i
		self.LS = {}
		for key in activeLearner.keys():
			result = [0,0,0,0]
			for i in range(len(result)):
				result[i] = activeLearner[key][i] * learner[0] + reflectiveLearner[key][i] * learner[1]
			self.LS[key] = result

	def getAction(self, int spawn, int visible, int danger, defOnCooldown = False):

		situation = self.LS['CLEAR']
		if danger>=2:
			situation = self.LS['DANGER_2']
		elif danger==1:
			situation = self.LS['DANGER_1']
		elif visible>=5:
			situation = self.LS['VISIBLE_3']
		elif visible >=3:
			situation = self.LS['VISIBLE_2']
		elif visible >=1:
			situation = self.LS['VISIBLE_1']

		gaussian = list(situation)
		for i in range(len(gaussian)):
			gaussian[i] = gauss(situation[i], 20)
			gaussian[i] = math.floor(max(0, min(100, gaussian[i])))

		total = math.floor(sum(gaussian))
		if defOnCooldown:
			total -= gaussian[self.DEFEND]
		if total == 0:
			return self.WAIT

		ran = rand()%total

		if ran <= gaussian[self.WAIT]:
			action = self.WAIT
		elif ran < gaussian[self.SWIPE] + gaussian[self.WAIT]:
			action = self.SWIPE
		elif ran < gaussian[self.BEACON] + gaussian[self.SWIPE] + gaussian[self.WAIT]:
			action = self.BEACON
		else:
			action = self.DEFEND

		return action

	def getAcc(self, situation, num):
		return self.LS[situation][num]


reflectiveLearner = {
	'CLEAR' : [10, 0,90,0],
	'SPAWN' : [5, 0,95,0],
	'VISIBLE_1' : [0, 20,80,0],
	'VISIBLE_2' : [0, 25,75,0],
	'VISIBLE_3' : [0, 50,50,0],
	'DANGER_1' : [0, 40,40,20],
	'DANGER_2' : [0, 10,5,85],
	'SWIPE_ACC' : [30,10,20,25],
	'BEACON_ACC' : [15,35,25,30],
	'DEFEND_ACC' : [90,70,60,80]
}

activeLearner = {
	'CLEAR' : [20, 20,50,10],
	'SPAWN' : [10, 25,60,5],
	'VISIBLE_1' : [10, 80,10,0],
	'VISIBLE_2' : [5, 85,10,0],
	'VISIBLE_3' : [10, 85,5,0],
	'DANGER_1' : [0, 80,0,20],
	'DANGER_2' : [0, 30,0,70],
	'SWIPE_ACC' : [40,20,30,35],
	'BEACON_ACC' : [10,30,20,25],
	'DEFEND_ACC' : [85,60,50,80]
}
