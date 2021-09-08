import argparse
import math
import numpy as np
from numpy import random

def pullArm(In,arm):
    sample=random.rand()
    for i in range(0,len(In[arm])):
        if(sample<In[arm][i]):
            break
    reward=In[0][i]
    return reward

def eGreedy(In,ep,hz):
    rewards=[[] for i in range(len(In)-1)]
    TotalReward=0
    avgrewards=[0 for i in range(len(In)-1)]
    sumrewards=[0 for i in range(len(In)-1)]
    pulls=[1 for i in range(len(In)-1)]
    for t in range(hz):
        if random.rand()<ep :
            arm=random.choice(range(1,len(In)))
        else:
            arm=avgrewards.index(max(avgrewards))+1
        rew=pullArm(In,arm)
        rewards[arm-1].append(rew)
        TotalReward+=rew
        sumrewards[arm-1]+=rew
        pulls[arm-1]+=1
        avgrewards[arm-1]=sumrewards[arm-1]/pulls[arm-1]
    return TotalReward

def UCB(In,hz,c=2):
    rewards=[[] for i in range(len(In)-1)]
    TotalRew=0
    avgrewards=[0 for i in range(len(In)-1)]
    sumrewards=[0 for i in range(len(In)-1)]
    pulls=[0 for i in range(len(In)-1)]
    for i in range(len(In)-1): #initialize
        rew=pullArm(In,i+1)
        rewards[i].append(rew)
        TotalRew+=rew
        pulls[i]+=1
        sumrewards[i]+=rew
        avgrewards[i]=sumrewards[i]/pulls[i]
    if hz<len(In)-1:
        print("Hz too short for ucb")
    for t in range(len(In),hz+1):
        ucbs=[avgrewards[arm]+math.sqrt(c*math.log(t)/pulls[arm]) for arm in range(len(pulls))]
        arm=ucbs.index(max(ucbs))+1
        rew=pullArm(In,arm)
        rewards[arm-1].append(rew)
        TotalRew+=rew
        sumrewards[arm-1]+=rew
        pulls[arm-1]+=1
        avgrewards[arm-1]=sumrewards[arm-1]/pulls[arm-1]
    return TotalRew

def KlUCB(In,hz):
    rewards=[[] for i in range(len(In)-1)]
    TotalRew=0
    avgrewards=[0 for i in range(len(In)-1)]
    sumrewards=[0 for i in range(len(In)-1)]
    pulls=[0 for i in range(len(In)-1)]
    for i in range(len(In)-1): #initialize
        rew=pullArm(In,i+1)
        rewards[i].append(rew)
        TotalRew+=rew
        pulls[i]+=1
        sumrewards[i]+=rew
        avgrewards[i]=sumrewards[i]/pulls[i]
    if hz<len(In)-1:
        print("Hz too short for ucb")
    for t in range(len(In),hz+1):
        if(t%100==0):
            print(t)
        ucbs=[]
        for arm in range(len(pulls)):
            p=avgrewards[arm]
            start=p
            end=1.0
            u=pulls[arm]
            c=5.0
            for itr in range(20):
                mid=(start+end)/2
                if p!=0 and p!=1:
                    term=p*math.log(p/mid)+(1-p)*math.log((1-p)/(1-mid))
                else:
                    term=0
                if u*(term)<math.log(t)+c*math.log(math.log(t)):
                    start=mid
                else:
                    end=mid
            ucbs.append(mid)     
        arm=ucbs.index(max(ucbs))+1
        rew=pullArm(In,arm)
        rewards[arm-1].append(rew)
        TotalRew+=rew
        sumrewards[arm-1]+=rew
        pulls[arm-1]+=1
        avgrewards[arm-1]=sumrewards[arm-1]/pulls[arm-1]
    return TotalRew

def ThompsonSampling(In,hz):
    rewards=[[] for i in range(len(In)-1)]
    sumrewards=[0 for i in range(len(In)-1)]
    pulls=[0 for i in range(len(In)-1)]
    TotalRew=0
    for t in range(hz):
        samples=[]
        for arm in range(len(pulls)):
            s=sumrewards[arm]
            f=pulls[arm]-s
            samples.append(random.beta(s+1,f+1))
        arm=samples.index(max(samples))+1
        rew=pullArm(In,arm)
        sumrewards[arm-1]+=rew
        pulls[arm-1]+=1
        rewards[arm-1].append(rew)
        TotalRew+=rew
    return TotalRew   

parser=argparse.ArgumentParser()
parser.add_argument('--instance')
parser.add_argument('--algorithm')
parser.add_argument('--randomSeed')
parser.add_argument('--epsilon')
parser.add_argument('--scale')
parser.add_argument('--threshold')
parser.add_argument('--horizon')

args = parser.parse_args()

In=args.instance
al=args.algorithm 
rs=int(args.randomSeed)
ep=float(args.epsilon)
c=float(args.scale)
th=float(args.threshold)
hz=int(args.horizon)

pathtoIn=In

random.seed(rs)
with open(In) as f:
    In=[]
    bern=0
    for a in f.readlines():
        if len(a.split())==1:
            bern=1
    f.seek(0)
    if(bern):
        In.append([0,1])
        for a in f.readlines():
            In.append([1-float(a),float(a)])
    else:
        for a in f.readlines():
            In.append([float(p) for p in a.split() ])

highs=0
if args.algorithm=='epsilon-greedy-t1':
    reward=eGreedy(In,ep,hz)
elif args.algorithm=='ucb-t1':
    reward=UCB(In,hz)
elif args.algorithm=='kl-ucb-t1':
    reward=KlUCB(In,hz)
elif args.algorithm=='thompson-sampling-t1':
    reward=ThompsonSampling(In,hz)
elif args.algorithm=='ucb-t2':
    reward=UCB(In,hz,c)
elif args.algorithm=='alg-t3':
    reward=ThompsonSampling(In,hz)
elif args.algorithm=='alg-t4':
    reward=ThompsonSampling(In,hz)
    if(reward>th):
        highs=1
maxExp=0 
for arm in range(1,len(In)):
    expRew=0
    for i in range(len(In[0])):
        expRew+=In[0][i]*In[arm][i]
    maxExp=max(expRew,maxExp)

regret=round(maxExp*hz-reward,3)

print(pathtoIn,al,rs,ep,c,th,hz,regret,highs,sep=", ")