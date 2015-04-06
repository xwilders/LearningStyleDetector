#from random import gauss
import math
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand

class Learner:
	
	NONE = -1
	WAIT = 0
	SWIPE = 1
	BEACON = 2
	DEFEND = 3

	DEFEND_TIME = 1.5
	DEFEND_COOLDOWN = 3
	BEACON_TIME = 3

	def __init__(self, learner):
		cdef int i
		self.LS = {}
		for key in activeLearner.keys():
			#Gaussian variance of 10
			self.LS[key] = activeLearner[key] * learner[0] + reflectiveLearner[key] * learner[1] + 10 * np.random.randn(4)
			self.LS[key] = self.LS[key]/sum(self.LS[key]) #Normalise

	def getAction(self, int spawn, int visible, int danger, bint defOnCooldown = False):

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

		cdef int total = 100
		if defOnCooldown:
			total -= situation[self.DEFEND]
		if total == 0:
			return self.WAIT

		cdef float ran = rand()%total
		cdef int action

		if ran <= situation[self.WAIT]:
			action = self.WAIT
		elif ran < situation[self.SWIPE] + situation[self.WAIT]:
			action = self.SWIPE
		elif ran < situation[self.BEACON] + situation[self.SWIPE] + situation[self.WAIT]:
			action = self.BEACON
		else:
			action = self.DEFEND

		return action

	def getAcc(self, situation, int num):
		return self.LS[situation][num]


reflectiveLearner = {
	'CLEAR' : np.array([0, 0,0,0]),
	'SPAWN' : np.array([5, 0,95,0]),
	'VISIBLE_1' : np.array([15, 5, 80,0]),
	'VISIBLE_2' : np.array([10, 30, 60,0]),
	'VISIBLE_3' : np.array([5, 65, 30,0]),
	'DANGER_1' : np.array([0, 50, 0,50]),
	'DANGER_2' : np.array([0, 5, 0,95]),
	'SWIPE_ACC' : np.array([75,25,40,40]),
	'BEACON_ACC' : np.array([30,50,40,40])
}

activeLearner = {
	'CLEAR' : np.array([0,0,0,0]),
	'SPAWN' : np.array([5, 5,85,5]),
	'VISIBLE_1' : np.array([5, 80,15,0]),
	'VISIBLE_2' : np.array([5, 85,10,0]),
	'VISIBLE_3' : np.array([0, 100,0,0]),
	'DANGER_1' : np.array([0, 50,0,50]),
	'DANGER_2' : np.array([0, 5,0,95]),
	'SWIPE_ACC' : np.array([100,40,60,60]),
	'BEACON_ACC' : np.array([25,45,35,35])
}
