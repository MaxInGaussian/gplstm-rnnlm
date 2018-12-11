
if [ "${3}" == "" ]; then

    nohup python main.py --ppl --dataset ${1}\
        --only-ngram --cuda ${2} > results/${1}-4gram.ppl &
        tail -f results/${1}-4gram.ppl

else

    if [ "${4}" == "" ]; then

        nohup python main.py --ppl --dataset ${1} --model ${2}\
            --load trained/${1}-${2} --cuda ${3} > results/${1}-${2}.ppl &
        tail -f results/${1}-${2}.ppl

    else

        nohup python main.py --ppl --dataset ${1} --model ${2}\
            --load trained/${1}-${2} --ngram --ngramscale ${3}\
            --cuda ${4} > results/${1}-${2}-4gram.ppl &
        tail -f results/${1}-${2}-4gram.ppl

    fi

fi