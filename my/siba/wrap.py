import os,random,subprocess
import matplotlib.pyplot as plt
import numpy as np

fname= "outputData.txt"

algo_ls_t1 = ['epsilon-greedy-t1','ucb-t1','kl-ucb-t1','thompson-sampling-t1']
algo_ls_t2 = 'ucb-t2'
algo_ls_t3 = 'alg-t3'
algo_ls_t4 = 'alg-t4'

inst_t1=['../instances/instances-task1/i-1.txt', '../instances/instances-task1/i-2.txt', '../instances/instances-task1/i-3.txt']
inst_t2=['../instances/instances-task2/i-1.txt', '../instances/instances-task2/i-2.txt', '../instances/instances-task2/i-3.txt', '../instances/instances-task2/i-4.txt', '../instances/instances-task2/i-5.txt']
inst_t3=['../instances/instances-task3/i-1.txt', '../instances/instances-task3/i-2.txt']
inst_t4=['../instances/instances-task4/i-1.txt', '../instances/instances-task4/i-2.txt']

horizon_t1=[100, 400, 1600, 6400, 25600, 102400]
horizon_t2=10000
horizon_t3=[100, 400, 1600, 6400, 25600, 102400]
horizon_t4=[100, 400, 1600, 6400, 25600, 102400]

randomseed=[i for i in range(50)]

scale=[0.02*i for i in range(1,16)]

ep=0.02

th=[0.2,0.6]

f = open(fname, "w")

#-------------------------------------------------------------------------------------Task1--------------------------------------------------------------------------------------------------------------
print("Task1 executing")
cnt=0
y=[[[0 for i in range(len(horizon_t1))] for j in range(len(algo_ls_t1))] for k in range(len(inst_t1))]
for inst in range(len(inst_t1)):
    print(inst)
    for algo in range(len(algo_ls_t1)):
        for hz in range(len(horizon_t1)):
            for rs in randomseed:
                cnt+=1
                cmd = "python","bandit.py","--instance",inst_t1[inst],"--algorithm",algo_ls_t1[algo],"--randomSeed",str(rs),"--epsilon",str(ep),"--scale","2","--threshold","0","--horizon",str(horizon_t1[hz])
                reproduced_str = subprocess.check_output(cmd,universal_newlines=True)
                f.write(reproduced_str.strip())
                f.write("\n")
                reproduced = reproduced_str.replace("\n","").split(",")
                REG = float(reproduced[-2].strip())
                y[inst][algo][hz]+=REG
            y[inst][algo][hz]=y[inst][algo][hz]/len(randomseed)


for inst in range(len(inst_t1)):
    print(inst)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for algo in range(len(algo_ls_t1)):
        line, = ax.plot(horizon_t1, y[inst][algo][:], label=algo_ls_t1[algo])
    ax.set_xscale('log',base=2)
    plt.legend(loc=1)
    plt.title("Task 1, Instance-"+str(inst+1))
    plt.xlabel('Horizon (log scale)')
    plt.ylabel('Average Regret over 50 runs')
    plt.savefig("Task1_inst"+str(inst+1)+".pdf")

print(cnt)

#-------------------------------------------------------------------------------------Task2--------------------------------------------------------------------------------------------------------------
print("Task2 executing")
y=[[0 for i in range(len(scale))] for j in range(len(inst_t2))]
for inst in range(len(inst_t2)):
    print(inst)
    for c in range(len(scale)):
        for rs in randomseed:
            cmd = "python","bandit.py","--instance",inst_t2[inst],"--algorithm",algo_ls_t2,"--randomSeed",str(rs),"--epsilon",str(ep),"--scale",str(scale[c]),"--threshold","0","--horizon",str(horizon_t2)
            reproduced_str = subprocess.check_output(cmd,universal_newlines=True)
            f.write(reproduced_str.strip())
            f.write("\n")
            reproduced = reproduced_str.replace("\n","").split(",")
            REG = float(reproduced[-2].strip())
            y[inst][c]+=REG
        y[inst][c]=y[inst][c]/len(randomseed)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for inst in range(len(inst_t2)):
    print(inst)
    line, = ax.plot(scale, y[inst][:], label="Inst-"+str(inst+1))

plt.xlabel('Scale')
plt.title("Task 2")
plt.ylabel('Average Regret over 50 runs')
plt.legend(loc=1)
plt.savefig("Task2"+".pdf")

#-------------------------------------------------------------------------------------Task3--------------------------------------------------------------------------------------------------------------
print("Task3 executing")
y=[[0 for i in range(len(horizon_t3))] for k in range(len(inst_t3))]
for inst in range(len(inst_t3)):
    print(inst)
    for hz in range(len(horizon_t3)):
        for rs in randomseed:
            cmd = "python","bandit.py","--instance",inst_t3[inst],"--algorithm",algo_ls_t3,"--randomSeed",str(rs),"--epsilon",str(ep),"--scale","2","--threshold","0","--horizon",str(horizon_t3[hz])
            reproduced_str = subprocess.check_output(cmd,universal_newlines=True)
            f.write(reproduced_str.strip())
            f.write("\n")
            reproduced = reproduced_str.replace("\n","").split(",")
            REG = float(reproduced[-2].strip())
            y[inst][hz]+=REG
        y[inst][hz]=y[inst][hz]/len(randomseed)


for inst in range(len(inst_t3)):
    print(inst)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    line, = ax.plot(horizon_t3, y[inst][:])
    ax.set_xscale('log',base=2)
    plt.title("Task 3, Instance-"+str(inst+1))
    plt.xlabel('Horizon (log scale)')
    plt.ylabel('Average Regret over 50 runs')
    plt.savefig("Task3_inst"+str(inst+1)+".pdf")


#-------------------------------------------------------------------------------------Task4--------------------------------------------------------------------------------------------------------------
print("Task4 executing")
y=[[[0 for i in range(len(horizon_t4))] for j in range(len(th))] for k in range(len(inst_t4))]
for inst in range(len(inst_t4)):
    print(inst)
    for t in range(len(th)):
        for hz in range(len(horizon_t4)):
            for rs in randomseed:
                cmd = "python","bandit.py","--instance",inst_t4[inst],"--algorithm",algo_ls_t4,"--randomSeed",str(rs),"--epsilon",str(ep),"--scale","2","--threshold",str(th[t]),"--horizon",str(horizon_t4[hz])
                reproduced_str = subprocess.check_output(cmd,universal_newlines=True)
                f.write(reproduced_str.strip())
                f.write("\n")
                reproduced = reproduced_str.replace("\n","").split(",")
                REG = float(reproduced[-2].strip())
                y[inst][t][hz]+=REG
            y[inst][t][hz]=y[inst][t][hz]/len(randomseed)


for inst in range(len(inst_t4)):
    print(inst)
    for t in range(len(th)):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        line, = ax.plot(horizon_t4, y[inst][t][:])
        ax.set_xscale('log',base=2)
        plt.title("Task 4, Instance-"+str(inst+1)+", Threshold-"+str(th[t]))
        plt.xlabel('Horizon (log scale)')
        plt.ylabel('Average Regret over 50 runs')
        plt.savefig("Task4_inst"+str(inst+1)+"_th-"+str(th[t])+".pdf")


f.close()