import argparse
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument("--policy","-p")
parser.add_argument("--states","-s")

args = parser.parse_args()

def get_states(filename):
	states = None
	with open(filename,"r") as f:
		states = [l.strip("\n") for l in f]

	states.append(len(states))

	return states

def next_state(state,action,opp,end):
	state = list(state)
	state[action] = opp
	state = ''.join(state)
	s = state

	win = 1

	for i in range(3):
		if(s[i] == s[i+3] and s[i] == s[i+6] and s[i]!='0'):
			return (win,end)

	for i in [0,3,6]:
		if(s[i] == s[i+1] and s[i+1] == s[i+2] and s[i]!='0'):
			return (win,end)

	if(s[0] == s[4] and s[0] == s[8] and s[0]!='0'):
		return (win,end)

	if(s[2] == s[4] and s[2] == s[6] and s[2]!='0'):
		return (win,end)


	for x in list(state):
		if(x == '0'):
			win = 0
	
	if(win==1):
		return (0,end)

	return (win,s)

def get_transitions(filename,end):
	pol = {}
	opponent = None
	with open(filename,"r") as f:
		for line in f:
			l = line.strip("\n").split(" ")
			if(l[0] == '1' or l[0] == '2'):
				opponent = l[0]
			else:
				# print(next_state(l[0],1,opponent))

				val = l[1:]

				pol[l[0]] = []
				for i in range(len(val)):
					if(float(val[i]) != 0):
						win,s = next_state(l[0],i,opponent,end)
						if(s == end):
							if(win == 0):
								pol[l[0]].append((s,float(val[i]),0)) ##if draw reward  = 0
							else:
								pol[l[0]].append((s,float(val[i]),1))
						else:
							pol[l[0]].append((s,float(val[i]),0))


						

	return opponent, pol



states = get_states(args.states)
end = len(states)-1

opponent, policy = get_transitions(args.policy,end)

player = None

if(opponent == '1'):
	player ='2'
else:
	player = '1'

print("numStates",len(states))
print("numActions",9)
print("end",len(states)-1)


for i in range(0,len(states)-1):
	curr = states[i]
	for a in range(9):
		trn = {}
		if(curr[a] == '0'):
			win,nextst = next_state(curr,a,player,end)
			# print(nextst)
			if(nextst == end):
				if(win == 0):
					print("transition",i,a,end,0,1.0)
				else:	
					print("transition",i,a,end,0,1.0)
			else:
				pol = policy[nextst]
				dr = 0
				for x,y,z in pol:
					if(x == end and z==0):
						dr = 1
					if(x in trn):
						trn[x] += y
					else:
						trn[x] = y

				for key in trn:
					idx = states.index(key)
					if(idx == end and dr==0):
						print("transition",i,a,end,1,trn[key])
					else:
						print("transition",i,a,idx,0,trn[key])
		else:
			print("transition",i,a,end,-1,1)


print("mdptype","episodic")
print("discount",1.0)












