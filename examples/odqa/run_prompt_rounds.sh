
#!/bin/bash
# gpu='4'


for i in `seq 57 64` 
    do
        bash examples/odqa/run_prompt.sh $RANDOM
        sleep 10
    done

# list='4 49'
# list='24 39'
# list='61 22'
# list='32'
# list='18 15'
# list='33'

# for i in ${list}
#     do
#         bash examples/odqa/run_prompt.sh $RANDOM rnd${i} ${gpu}
#         sleep 10
#     done