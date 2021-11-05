import argparse
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument("--value-policy","-v")
parser.add_argument("--states","-s")
parser.add_argument("--player-id","-pl")

args = parser.parse_args()

def get_states(filename):
	states = None
	with open(filename,"r") as f:
		states = [l.strip("\n") for l in f]

	states.append(len(states))

	return states

def get_policy(filename):
	pi = None
	with open(filename,"r") as f:
		pi = [int(l.split()[1]) for l in f]

	return pi


states = get_states(args.states)

player = args.player_id

print(player)

pi = get_policy(args.value_policy)

for i in range(0,len(states)-1):
	pol = [float(0) for j in range(9)]
	pol[pi[i]] = 1.0
	print(states[i],pol[0],pol[1],pol[2],pol[3],pol[4],pol[5],pol[6],pol[7],pol[8])











