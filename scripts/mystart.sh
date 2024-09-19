#!/bin/bash
#${iv[0]}
max_comm=1
population=100 #100 #4
iterations=52 #2 #52

seeds=(5) #(1 6 11 16 21 26) #(15 18) #(1 11 21) #($(seq 2 2 )) #($(seq 1 30 ))
topology=("CONV") #("DENSE" "CONV")
num_layers=(12) #(4 12)
speeds=("0.5,0.75,1.0,2.0") # ("1.0,1.0,1.0,1.0") #("0.333333,0.5,0.666666,1.0") #("1.0,1.0,1.0,1.0" "3.0,2.0,1.5,1.0") #("3.0,2.0,1.5,1.0") #("1.0,1.0,1.0,1.0") #("1.0,1.0,1.0,1.0") # ("1.0,1.0,1.0,1.0" "3.0,2.0,1.5,1.0")  #("3.0,3.0,3.0,3.0") #("1.0,1.0,1.0,1.0" "3.0,3.0,1.5,1.0" "3.0,1.5,1.5,1.0" "3.0,1.5,1.0,1.0", "3.0,3.0,3.0,3.0")
num_devices=4


## INCREMENTAL START
#incr_values=((1.0 0 999) (0.4 0.2 5) (0.1 0.15 3))
incr_init=(1.0) #(1.0) # (1.0 0.4 0.1)
incr_amount=(0) #(0 0.2 0.15)
incr_gen=(999) #(999 5 3)
repeat_entire_pop=(0) #(1 0)
####################

## MYMUTINT
mutation=("mutUniformInt") #("mutUniformInt") #("myMutInt") #("mutUniformInt") #("myMutInt" "mutUniformInt")
decrement_amount_prob=(2) #(2 4 10) 
####################

strategy_type=("EDA") #("NONE" "EDA" "CMAES")


dataset=("FASHION") #("MNIST" "FASHION") #("MNIST" "FASHION") #("MNIST" "FASHION") #("MNIST","FASHION")

for seed in "${seeds[@]}"
do
  for rep in "${repeat_entire_pop[@]}"
  do
    for (( i=0; i<${#topology[@]}; i++ )); #for top in "${topology[@]}"
    do
      for sp in "${speeds[@]}"
      do
        for (( j=0; j<${#incr_init[@]}; j++ ));
        do
          for mut in "${mutation[@]}"
          do
            for dat in "${dataset[@]}"
            do     
              for decr in "${decrement_amount_prob[@]}"
              do    
                for st in "${strategy_type[@]}"
                do    
                  name="${dat:0:1}_${topology[$i]:0:1}_${seed}_${st:0:1}_${mut:0:5}_${decr}_${sp}_${incr_init[$j]}_${incr_amount[$j]}_${incr_gen[$j]}_${rep}_${max_comm}"
                  echo $name
                  echo $dat
                  #./start.sh $name -s $seed -i $iterations -p $population -t ${topology[$i]} -l ${num_layers[$i]} -d $num_devices -sp $sp -ii ${incr_init[$j]} -ia ${incr_amount[$j]}  -ig ${incr_gen[$j]} -r $rep -maxc $max_comm -m $mut -decr $decr -dt $dat -st $st
                  sbatch slurm_with_params.sh -J $name -s $seed -i $iterations -p $population -t ${topology[$i]} -l ${num_layers[$i]} -d $num_devices -sp $sp -ii ${incr_init[$j]} -ia ${incr_amount[$j]}  -ig ${incr_gen[$j]} -r $rep -maxc $max_comm -m $mut -decr $decr -dt $dat -st $st    
                  #./start.sh $name -s $seed -i $iterations -p $population -t ${topology[$i]} -l ${num_layers[$i]} -d $num_devices -sp $sp -ii ${incr_init[$j]} -ia ${incr_amount[$j]}  -ig ${incr_gen[$j]} -r $rep -maxc $max_comm -m $mut -decr $decr -dt $dat -st $st
                  #exit 0
                  sleep 5                  
                done                          
              done                                            
            done                
          done
        done
      done
    done
  done
done


rm slurm-*

#sacct --format="JobID,JobName%100"

