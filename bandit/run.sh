instances=("../instances/instances-task1/i-1.txt" "../instances/instances-task1/i-2.txt" "../instances/instances-task1/i-3.txt")
algos=("epsilon-greedy-t1" "ucb-t1" "kl-ucb-t1" "thompson-sampling-t1")
horizons=(100 400 1600 6400 25600 102400)
for instance in ${instances[@]}
do
    for algo in ${algos[@]}
    do
        for horizon in ${horizons[@]}
        do
            for seed in {0..49}
            do
                echo $algo,$horizon,$seed
                python3 bandit.py --instance $instance --algorithm $algo --randomSeed $seed --epsilon 0.02 --scale 2 --threshold 0 --horizon $horizon >> newdata1.txt
            done
        done
    done
done