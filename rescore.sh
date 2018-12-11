
if [ "${3}" == "" ]; then

    nohup python main.py --rescore --dataset ${1} --lmscale ${2} --save results/${1}-4gram.${2}.hyp\
        --only-ngram > results/${1}-4gram.${2}.rescore &

else

    if [ "${5}" == "" ]; then

        nohup python main.py --rescore --dataset ${1} --lmscale ${3} --save results/${1}-${2}.${3}.hyp\
            --model ${2} --load trained/${1}-${2} --cuda ${4} > results/${1}-${2}.${3}.rescore &

    else

        nohup python main.py --rescore --dataset ${1} --lmscale ${3} --save results/${1}-${2}-4gram.${3}.hyp\
            --model ${2} --load trained/${1}-${2} --ngram --ngramscale ${5} --cuda ${4} > results/${1}-${2}-4gram.${3}.rescore &

    fi

fi