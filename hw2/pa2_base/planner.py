import argparse
import numpy as np 
from pulp import *
from sparse_matrix import *

neginf = -1e60

def get_newv(s,trans,reward,v,gamma):
	newv = [neginf for i in range(s)]
	for key in trans.mat:
		if(newv[key[0]] == neginf):
			newv[key[0]] = 0
		newv[key[0]] += trans.mat[key]*(reward.mat[key]+gamma*v[key[1]])
	return newv


def vi(s,a,e,tr,rew,mdp,gamma):
	v = [0 for i in range(s)]
	newv = []
	lastv = [1000 for i in range(s)]
	v = np.array(v)
	lastv = np.array(lastv)

	while((v == lastv).all() == False):
		newv = [get_newv(s,tr[j],rew[j],v,gamma) for j in range(a)]
		newv = np.array(newv)
		lastv = v
		v = np.max(newv,axis=0)
		for ep in e:
			v[ep] = 0

	pi = np.argmax(newv,axis=0)
	for ep in e:
		pi[ep] = 0
	return v,pi

def get_rhs(i,vals):
	rhs = 0
	for op in vals:
		rhs += op[1]*op[2]
	return rhs

def compute_lhs_vector(i,s,vals,gamma):
	vec = np.zeros(s)
	for op in vals:
		vec[op[0]] -= gamma*op[1]
	vec[i] += 1
	return vec

def computeV(s,pi,tr,rew,gamma,dc):
	b = np.array([get_rhs(i,dc.get((i,pi[i]),[])) for i in range(s)])
	X = np.array([compute_lhs_vector(i,s,dc.get((i,pi[i]),[]),gamma) for i in range(s)])
	q = np.linalg.solve(X,b)
	return q 

def computeq(vals,gamma,v):
	q = 0
	for op in vals:
		q += op[1]*(op[2] + gamma*v[op[0]])
	return q

def policyImprovement(s,a,e,tr,rew,mdp,gamma,pi,dc):
	v = computeV(s,pi,tr,rew,gamma,dc)
	newpi = pi
	changed = 0

	for i in range(s):
		for j in range(a):
			if(computeq(dc.get((i,j),[]),gamma,v)-v[i]> 1e-7):
				newpi[i] = j
				changed = 1
				break

	return newpi,changed,v


def hpi(s,a,e,tr,rew,mdp,gamma):
	pi = [0 for i in range(s)]
	dc = {}
	for i in range(a):
		for key in tr[i].mat:
			if (key[0],i) not in dc.keys():
				dc[key[0],i] = []
			dc[key[0],i].append((key[1],tr[i].mat[key],rew[i].mat[key]))

	for i in range(s):
		for j in range(a):
			if (i,j) in dc.keys():
				pi[i] = j
				break

	v = None

	while(1): 
		pi,changed,v = policyImprovement(s,a,e,tr,rew,mdp,gamma,pi,dc)
		if(changed == 0):
			break

	for ep in e:
		pi[ep] = 0
	return v,pi


def lp(S,a,e,tr,rew,mdp,gamma):
	prob = LpProblem("mdp",LpMinimize)
	v = [ LpVariable("v"+str(i)) for i in range(S)]
	prob += lpSum(v)

	for i in range(a):
		lineq = [[] for j in range(S)]

		for key in tr[i].mat:
			lineq[key[0]].append(tr[i].mat[key]*(rew[i].mat[key] + gamma*v[key[1]]))

		for j in range(S):
			if(lineq[j] != []):
				prob += (
					v[j] >= lpSum(lineq[j])
					)

	for s in e:
		prob += (v[s] == 0)


	# print(prob)
	prob.solve(PULP_CBC_CMD(msg=0))

	v = [value(v[s]) for s in range(S)]
	newv = [get_newv(S,tr[j],rew[j],v,gamma) for j in range(a)]
	newv = np.array(newv)
	pi = np.argmax(newv,axis=0)

	return v,pi



parser = argparse.ArgumentParser()
parser.add_argument("--mdp","-m")
parser.add_argument("--algorithm","-al", default = "lp")

args = parser.parse_args()

file = args.mdp

s = 0 
a = 0
e = []
mdp = ""
gamma = 1
rew = None
tr = None

with open(file,'r') as f:
	for line in f:
		l = line.split()

		if(l[0] == "numStates"):
			s = int(l[1])
		elif(l[0] == "numActions"):
			a = int(l[1])
			tr = [sparse_matrix(s,s) for i in range(a)]
			rew = [sparse_matrix(s,s) for i in range(a)]

		elif(l[0] == "end"):
			if(int(l[1]) != -1):
				e = [int(x) for x in l[1:]]

		elif(l[0] == "transition"):
			tr[int(l[2])].addelem(int(l[1]),int(l[3]),float(l[5]))
			rew[int(l[2])].addelem(int(l[1]),int(l[3]),float(l[4]))

		elif(l[0] == "mdptype"):
			mdp = l[1]

		elif(l[0] == "discount"):
			gamma = float(l[1])

if(args.algorithm == "vi"):
	v,pi = vi(s,a,e,tr,rew,mdp,gamma)

elif(args.algorithm == "hpi"):
	v,pi = hpi(s,a,e,tr,rew,mdp,gamma)

elif(args.algorithm == "lp"):
	v,pi = lp(s,a,e,tr,rew,mdp,gamma)


for i in range(s):
	print("{0:0.6f}".format(v[i]),pi[i])