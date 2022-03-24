
#!/bin/bash

# random_seed='2345 3456 4567 5678 6789 7890'


for i in `seq 57 64` 
    do
        bash examples/odqa/run_prompt.sh $RANDOM rnd${i} 
        sleep 10
    done