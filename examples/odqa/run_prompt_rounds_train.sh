
#!/bin/bash
# gpu='6'


for i in {17..24..1} 
    do
        echo ${i}
        nohup bash examples/odqa/run_prompt_train.sh rnd${i}
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