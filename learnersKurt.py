from random import random, gauss
import math

class Learner:
	
	WAIT = 0
	SWIPE = 1
	BEACON = 2
	DEFEND = 3

	DEFEND_TIME = 1
	DEFEND_COOLDOWN = 2
	BEACON_TIME = 3

	def __init__(self, dicts):
		self.dicts = dicts		

	def getSit(self, situation, learner):
		result = [0,0,0,0]
		for i in range(len(result)):
			result[i] += reflectiveLearner[situation][i] * learner[0] + activeLearner[situation][i] * learner[1]
		return result

	def getAction(self, learner, spawn, visible, danger, defOnCooldown = False):

		situation = self.getSit('CLEAR', learner)
		if danger>=2:
			situation = self.getSit('DANGER_2', learner)
		elif danger==1:
			situation = self.getSit('DANGER_1', learner)
		elif visible>=5:
			situation = self.getSit('VISIBLE_3', learner)
		elif visible >=3:
			situation = self.getSit('VISIBLE_2', learner)
		elif visible >=1:
			situation = self.getSit('VISIBLE_1', learner)

		gaussian = list(situation)
		for i in range(len(gaussian)):
			gaussian[i] = gauss(situation[i], 20)
			gaussian[i] = max(0, min(100, gaussian[i]))

		total = sum(gaussian)
		if defOnCooldown:
			total -= gaussian[self.DEFEND]
		ran = random()*total

		if ran <= gaussian[self.WAIT]:
			action = self.WAIT
		elif ran < gaussian[self.SWIPE] + gaussian[self.WAIT]:
			action = self.SWIPE
		elif ran < gaussian[self.BEACON] + gaussian[self.SWIPE] + gaussian[self.WAIT]:
			action = self.BEACON
		elif ran < gaussian[self.DEFEND] + gaussian[self.BEACON] + gaussian[self.SWIPE] + gaussian[self.WAIT]:
			action = self.DEFEND
		return action

	def getAcc(self, learner, situation, num):
		return reflectiveLearner[situation][num]*learner[0] + activeLearner[situation][num]*learner[1]


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
