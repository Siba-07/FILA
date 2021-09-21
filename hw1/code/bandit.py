import argparse
import numpy as np 
from functools import partial
import math

def read_bandits(filename):
	f = open(filename, "r")
	bandits = []

	for x in f:
		bandits.append(float(x))

	bandits = np.array(bandits)

	return bandits

def read_fullbandits(filename):
	f = open(filename,"r")
	bandits = []
	rewardvals = []

	first = f.readline()

	for words in first.split():
		rewardvals.append(float(words))

	rewardvals = np.array(rewardvals)

	while True:
		line = f.readline()
		if not line:
			break

		instance = []

		for words in line.split():
			instance.append(float(words))

		bandits.append(instance)

	bandits = np.array(bandits)

	return bandits,rewardvals


def new_reward(mean,rewardvals):
	x = np.random.uniform(0,1)

	cum = 0
	for i in range(len(mean)):
		cum += mean[i]
		if(x<=cum):
			return rewardvals[i]

def bernoulli_reward(mean):
	x = np.random.uniform(0,1)

	if(x<mean):
		return 1
	else:
		return 0

def kl_div(x,y):
	return x*math.log(x)-x*math.log(y) + (1-x)*math.log(1-x) - (1-x)*math.log(1-y)

def binsearch(mean,time,th):
	l = mean 
	r = 1

	if(mean == 1 or mean == 0):
		return 1

	lim = 0.0001
	for i in range(20):
		mid = (l+r)/2
		if (time*kl_div(mean,mid) <= th):
			l = mid
		else: 
			r = mid
	return l

def algos(args):

	reg = 0
	rew = 0
	high = 0
	hz = int(args.horizon)

	if(args.algorithm == "alg-t3" or args.algorithm == "alg-t4"):
		bandits,rewardvals = read_fullbandits(args.instance)
		exprew = np.zeros(len(bandits))
		for i in range(len(bandits)):
			bandit = bandits[i]
			exp = 0
			for j in range(len(bandit)):
				exp += rewardvals[j]*bandit[j]
			exprew[i] = exp 

		best_arm = np.max(exprew)
		maximum_rew = hz*best_arm
	else:
		bandits = read_bandits(args.instance)
		best_arm = np.max(bandits)
		maximum_rew = hz*best_arm

	n = len(bandits)

	if(args.algorithm == "epsilon-greedy-t1"):
		eps = float(args.epsilon)
		times = np.zeros(n)
		empirical = np.zeros(n)

		for i in range(hz):
			way = np.random.uniform(0,1)

			choosen = -1

			if(way<eps):
				choosen = np.random.randint(n)
			else:
				choosen = np.argmax(empirical)

			reward = bernoulli_reward(bandits[choosen])
			rew += reward
			times[choosen] += 1
			empirical[choosen] = empirical[choosen]*(times[choosen]-1) + reward 
			empirical[choosen] = float(empirical[choosen])/float(times[choosen])
		
		reg = maximum_rew - rew 


	elif(args.algorithm == "ucb-t1" or args.algorithm=="ucb-t2"):
		times = np.zeros(n)
		empirical = np.zeros(n)

		c = float(args.scale)

		for i in range(n):
			choosen = i
			reward = bernoulli_reward(bandits[choosen])
			rew += reward
			times[choosen] += 1
			empirical[choosen] = empirical[choosen]*(times[choosen]-1) + reward 
			empirical[choosen] = empirical[choosen]/float(times[choosen])

		for i in range(hz-n):
			ucb = empirical + np.sqrt(((c*math.log(hz))/times))
			choosen = np.argmax(ucb)
			reward = bernoulli_reward(bandits[choosen])
			rew += reward
			times[choosen] += 1
			empirical[choosen] = empirical[choosen]*(times[choosen]-1) + reward 
			empirical[choosen] = empirical[choosen]/float(times[choosen])
		
		reg = maximum_rew - rew 

	elif(args.algorithm == "kl-ucb-t1"):
		times = np.zeros(n)
		empirical = np.zeros(n)
		sum_rewards = np.zeros(n)

		for i in range(n):
			choosen = i
			reward = bernoulli_reward(bandits[choosen])
			rew += reward
			sum_rewards[choosen] += reward 
			times[choosen] += 1
			empirical[choosen] = float(sum_rewards[choosen])/float(times[choosen])
 	
		for t in range(n,hz):
			ucb = []
			th = math.log(t) + 3*math.log(math.log(t))
			ucb = list(map(partial(binsearch,th=th),empirical,times))
			choosen = np.argmax(ucb)
			reward = bernoulli_reward(bandits[choosen])
			rew += reward
			sum_rewards[choosen] += reward 
			times[choosen] += 1
			empirical[choosen] = float(sum_rewards[choosen])/float(times[choosen])
			
		reg = maximum_rew - rew 


	elif(args.algorithm == "thompson-sampling-t1"):
		s = np.zeros(n)
		f = np.zeros(n)
		sample = np.zeros(n)

		for i in range(hz):
			for j in range(n):
				sample[j] = np.random.beta(s[j]+1,f[j]+1)

			choosen = np.argmax(sample)
			reward = bernoulli_reward(bandits[choosen])
			rew += reward

			if(reward == 1):
				s[choosen] += 1
			else:
				f[choosen] += 1
		
		reg = maximum_rew - rew 


	elif(args.algorithm == "alg-t3"):
		s = np.zeros(n)
		f = np.zeros(n)
		sample = np.zeros(n)
		times = np.zeros(n)
		empirical = np.zeros(n)

		for i in range(hz):
			for j in range(n):
				sample[j] = np.random.beta(s[j]+1,f[j]+1)

			choosen = np.argmax(sample)
			reward = new_reward(bandits[choosen],rewardvals)
			rew += reward

			
			s[choosen] += reward
			f[choosen] = f[choosen] + (1-reward)

		reg = maximum_rew - rew 

	elif(args.algorithm == "alg-t4"):
		s = np.zeros(n)
		f = np.zeros(n)
		sample = np.zeros(n)

		th = float(args.threshold)

		for i in range(hz):
			for j in range(n):
				sample[j] = np.random.beta(s[j]+1,f[j]+1)

			choosen = np.argmax(sample)
			reward = new_reward(bandits[choosen],rewardvals)
			rew += reward
			
			if(reward > th):
				s[choosen] += 1
			else:
				f[choosen] += 1

		for i in range(n):
			high += s[i]

	return reg,high


parser = argparse.ArgumentParser()

parser.add_argument("--instance","-in")
parser.add_argument("--algorithm","-al")
parser.add_argument("--randomSeed","-rs")
parser.add_argument("--epsilon","-ep")
parser.add_argument("--scale","-c")
parser.add_argument("--threshold","-th")
parser.add_argument("--horizon","-hz")

args = parser.parse_args()

np.random.seed(int(args.randomSeed))

reg,high = algos(args)

print(args.instance,args.algorithm,args.randomSeed,args.epsilon,args.scale,args.threshold,args.horizon,reg,high,sep=", ")



# print(args.instance,args.algorithm,args.randomSeed,args.epsilon,args.scale,args.threshold,args.horizon,args.reg,args.high,sep=", ")

	
