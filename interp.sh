
if [ "${6}" == "" ]; then

    python main.py --interp --dataset ${1} --ppl1 data/pma/${1}_pma/${2}.ppl --lambda1 ${3} --ppl2 data/pma/${1}_pma/${4}.ppl --save results/${1}-${2}-${4}.hyp --lmscale ${5}

else

    python main.py --interp --dataset ${1} --ppl1 data/pma/${1}_pma/${2}.ppl --lambda1 ${3} --ppl2 data/pma/${1}_pma/${4}.ppl --lambda2 ${5} --ppl3 data/pma/${1}_pma/${6}.ppl --save results/${1}-${2}-${4}-${6}.hyp --lmscale ${7}

fi