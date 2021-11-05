import random,argparse,sys,subprocess,os
import numpy as np

from numpy import cdouble
parser = argparse.ArgumentParser()

polfile = None

def run(states,policy,player,i,root):
	if(player == '1'):
		cmd_encoder = "python","encoder.py","--policy",policy,"--states",states
		f = open('mdp_player1','w')
		subprocess.call(cmd_encoder,stdout=f)
		f.close()

		cmd_planner = "python","planner.py","--mdp","mdp_player1"
		f = open('vp_player1','w')
		subprocess.call(cmd_planner,stdout=f)
		f.close()

		cmd_decoder = "python","decoder.py","--value-policy","vp_player1","--states",states ,"--player-id",str(player)
		f = open(root + '/policy_player1_{}'.format(int(i/2)),'w')
		subprocess.call(cmd_decoder,stdout=f)
		f.close()
		polfile = root + '/policy_player1_{}'.format(int(i/2))

		return polfile
	else:
		cmd_encoder = "python","encoder.py","--policy",policy,"--states",states
		f = open('mdp_player2','w')
		subprocess.call(cmd_encoder,stdout=f)
		f.close()

		cmd_planner = "python","planner.py","--mdp","mdp_player2"
		f = open('vp_player2','w')
		subprocess.call(cmd_planner,stdout=f)
		f.close()

		cmd_decoder = "python","decoder.py","--value-policy","vp_player2","--states",states ,"--player-id",str(player)
		f = open(root + '/policy_player2_{}'.format(int(i/2)),'w')
		subprocess.call(cmd_decoder,stdout=f)
		f.close()
		polfile = root + '/policy_player2_{}'.format(int(i/2))

		return polfile

def getPlayerId(policy):
    with open(policy,'r') as file:
        line = file.readline()
    opponent_player = line.strip()
    if opponent_player=='1':
        player = '2'
    else:
        player = '1'
    return player

def create_pol(policy,player,root):
	g = None
	if(player == '2'):
		g = open(root + '/policy_player1_0','w')
		polfile = root + '/policy_player1_0'
	else:
		g = open(root + '/policy_player2_0','w')
		polfile = root + '/policy_player2_0'

	with open(policy,'r') as f:
		for l in f:
			g.write(l)

	g.close()
	return polfile


def evaluate(lastfile,polfile):
	f = open(lastfile,'r')
	f.readline()
	newpol = []
	oldpol = []

	while(1):
		l = f.readline()
		if not l:
			break
		newpol.append([float(x) for x in l.strip().split()[1:]])

	f.close()
	f = open(polfile,'r')
	f.readline()

	while(1):
		l = f.readline()
		if not l:
			break
		oldpol.append([float(x) for x in l.strip().split()[1:]])

	count = 0

	f.close()

	for i in range(len(newpol)):
		x = newpol[i]
		y = oldpol[i]
		
		x = np.array(x)
		y = np.array(y)
		if(np.sum(x==y) != len(x)):
			count+=1

	print("New and old policies differ for " + str(count) + " states")


if __name__ == '__main__':
    # parser.add_argument("--s1",required=True,type=str,help="File with valid states of the player1")
    # parser.add_argument("--s2",required=True,type=str,help="File with valid states of the player2")
    # parser.add_argument("--policy",required=True,type=str,help="Policy file of the opponent player")
    # args = parser.parse_args()

    s1 = './data/attt/states/states_file_p1.txt'
    s2 = './data/attt/states/states_file_p2.txt'

    policies = ['./data/attt/policies/p1_policy1.txt','./data/attt/policies/p1_policy2.txt','./data/attt/policies/p2_policy1.txt','./data/attt/policies/p2_policy2.txt']

    os.system("mkdir task3files")

    for op in range(1):
    	policy = policies[op]
    	print("\nInitiating Learning for policy "+ policy)
    	player = getPlayerId(policy)
    	root = "./task3files/pol{}".format(op)
    	os.system("mkdir "+root)
    	polfile = create_pol(policy,player,root)
    	l1 = None
    	l2 = None
    	for i in range(1,21):
	    	if(player == '1'):
	    		print("iteration",int(i/2),": Player1 policy update")
	    		l2 = polfile
	    		polfile = run(s1,polfile,player,i,root)
	    		if l1 is not None:
	    			evaluate(l1,polfile)
	    		player = '2'
	    	else:
	    		print("iteration",int(i/2),": Player2 policy update")
	    		l1 = polfile
	    		polfile = run(s2,polfile,player,i,root)
	    		if l2 is not None:
	    			evaluate(l2,polfile)
	    		player = '1'








