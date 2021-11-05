import os

cmd1 = "python encoder.py --states ./data/attt/states/states_file_p{}.txt --policy ./data/attt/policies/p{}_policy{}.txt > mdp{}{}"
cmd2 = "python planner.py --mdp mdp{}{} > vp{}{}"
cmd3 = "python decoder.py --player-id {} --states ./data/attt/states/states_file_p{}.txt --value-policy vp{}{} > pol{}{}"

for i in [1,2]:
	for j in [1,2]:
		c1 = cmd1.format(i,i%2+1,j,i,j)
		c2 = cmd2.format(i,j,i,j)
		c3 = cmd3.format(i,i,i,j,i,j)

		os.system(c1)
		os.system(c2)
		os.system(c3)

		print("done for:",i,j)
	print("-------------------------------------")
    
