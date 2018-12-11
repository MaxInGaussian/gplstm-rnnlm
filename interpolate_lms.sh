# Reference Tutorial: http://www.speech.sri.com/projects/srilm/manpages/ppl-scripts.1.html


if [ "${3}" == "" ]; then

    # Interpolation of Ngram Model and RNN Language Model

    compute-best-mix 'data/'${1}'.ppl'\
        'trained/'${1}'-'${2}'.ppl'

    # sh interpolate_rnnlm.sh ptb lstm
    # iteration 1, lambda = (0.5 0.5), ppl = 101.4
    # iteration 2, lambda = (0.453777 0.546223), ppl = 100.646
    # iteration 3, lambda = (0.419533 0.580467), ppl = 100.23
    # iteration 4, lambda = (0.394326 0.605674), ppl = 100.001
    # iteration 5, lambda = (0.375792 0.624208), ppl = 99.8755
    # iteration 6, lambda = (0.362146 0.637854), ppl = 99.8062
    # iteration 7, lambda = (0.352078 0.647922), ppl = 99.7679
    # iteration 8, lambda = (0.344634 0.655366), ppl = 99.7467
    # iteration 9, lambda = (0.339118 0.660882), ppl = 99.7349
    # iteration 10, lambda = (0.335023 0.664977), ppl = 99.7284
    # iteration 11, lambda = (0.331981 0.668019), ppl = 99.7248
    # iteration 12, lambda = (0.329716 0.670284), ppl = 99.7228
    # iteration 13, lambda = (0.32803 0.67197), ppl = 99.7217
    # iteration 14, lambda = (0.326774 0.673226), ppl = 99.721
    # 82430 non-oov words, best lambda (0.325837 0.674163)

    # sh interpolate_rnnlm.sh ptb lstm-gpnn
    # iteration 1, lambda = (0.5 0.5), ppl = 99.6943
    # iteration 2, lambda = (0.446079 0.553921), ppl = 98.6837
    # iteration 3, lambda = (0.405949 0.594051), ppl = 98.1203
    # iteration 4, lambda = (0.376324 0.623676), ppl = 97.8068
    # iteration 5, lambda = (0.354478 0.645522), ppl = 97.6321
    # iteration 6, lambda = (0.338333 0.661667), ppl = 97.5343
    # iteration 7, lambda = (0.326363 0.673637), ppl = 97.4794
    # iteration 8, lambda = (0.317459 0.682541), ppl = 97.4485
    # iteration 9, lambda = (0.310815 0.689185), ppl = 97.4311
    # iteration 10, lambda = (0.305844 0.694156), ppl = 97.4212
    # iteration 11, lambda = (0.302118 0.697882), ppl = 97.4156
    # iteration 12, lambda = (0.29932 0.70068), ppl = 97.4125
    # iteration 13, lambda = (0.297216 0.702784), ppl = 97.4107
    # iteration 14, lambda = (0.295633 0.704367), ppl = 97.4096
    # iteration 15, lambda = (0.29444 0.70556), ppl = 97.409
    # 82430 non-oov words, best lambda (0.293541 0.706459)

    # sh interpolate_rnnlm.sh swbd lstm
    # iteration 1, lambda = (0.5 0.5), ppl = 69.3778
    # iteration 2, lambda = (0.490698 0.509302), ppl = 69.3559
    # iteration 3, lambda = (0.483045 0.516955), ppl = 69.3411
    # iteration 4, lambda = (0.476755 0.523245), ppl = 69.3311
    # iteration 5, lambda = (0.471587 0.528413), ppl = 69.3243
    # iteration 6, lambda = (0.467343 0.532657), ppl = 69.3198
    # iteration 7, lambda = (0.463858 0.536142), ppl = 69.3167
    # iteration 8, lambda = (0.460997 0.539003), ppl = 69.3146
    # iteration 9, lambda = (0.458648 0.541352), ppl = 69.3132
    # iteration 10, lambda = (0.45672 0.54328), ppl = 69.3122
    # iteration 11, lambda = (0.455138 0.544862), ppl = 69.3116
    # iteration 12, lambda = (0.453839 0.546161), ppl = 69.3112
    # iteration 13, lambda = (0.452772 0.547228), ppl = 69.3109
    # 45979 non-oov words, best lambda (0.451897 0.548103)

    # sh interpolate_rnnlm.sh swbd lstm-gpnn
    # iteration 1, lambda = (0.5 0.5), ppl = 69.1083
    # iteration 2, lambda = (0.490511 0.509489), ppl = 69.0856
    # iteration 3, lambda = (0.482725 0.517275), ppl = 69.0703
    # iteration 4, lambda = (0.476342 0.523658), ppl = 69.0601
    # iteration 5, lambda = (0.471112 0.528888), ppl = 69.0532
    # iteration 6, lambda = (0.466828 0.533172), ppl = 69.0486
    # iteration 7, lambda = (0.46332 0.53668), ppl = 69.0455
    # iteration 8, lambda = (0.460448 0.539552), ppl = 69.0434
    # iteration 9, lambda = (0.458096 0.541904), ppl = 69.042
    # iteration 10, lambda = (0.456171 0.543829), ppl = 69.041
    # iteration 11, lambda = (0.454595 0.545405), ppl = 69.0404
    # iteration 12, lambda = (0.453305 0.546695), ppl = 69.04
    # iteration 13, lambda = (0.452249 0.547751), ppl = 69.0397
    # 45979 non-oov words, best lambda (0.451384 0.548616)

else

    # Interpolation of Ngram Model and Two RNN Language Models

    compute-best-mix 'data/'${1}'.ppl'\
        'trained/'${1}'-'${2}'.ppl'\
        'trained/'${1}'-'${3}'.ppl'

    # sh interpolate_rnnlm.sh swbd lstm lstm-gpnn
    # iteration 1, lambda = (0.5 0.25 0.25), ppl = 67.8464
    # iteration 2, lambda = (0.479783 0.259286 0.260931), ppl = 67.7445
    # iteration 3, lambda = (0.462964 0.2669 0.270136), ppl = 67.6738
    # iteration 4, lambda = (0.449004 0.273122 0.277874), ppl = 67.625
    # iteration 5, lambda = (0.437435 0.278192 0.284373), ppl = 67.5913
    # iteration 6, lambda = (0.427853 0.282315 0.289832), ppl = 67.568
    # iteration 7, lambda = (0.419921 0.28566 0.294419), ppl = 67.552
    # iteration 8, lambda = (0.413354 0.288369 0.298277), ppl = 67.5409
    # iteration 9, lambda = (0.407918 0.290556 0.301526), ppl = 67.5332
    # iteration 10, lambda = (0.403417 0.292317 0.304266), ppl = 67.528
    # iteration 11, lambda = (0.39969 0.29373 0.30658), ppl = 67.5243
    # iteration 12, lambda = (0.396603 0.294859 0.308538), ppl = 67.5218
    # iteration 13, lambda = (0.394046 0.295756 0.310198), ppl = 67.5201
    # iteration 14, lambda = (0.391927 0.296465 0.311608), ppl = 67.5189
    # iteration 15, lambda = (0.390172 0.297021 0.312807), ppl = 67.518
    # iteration 16, lambda = (0.388717 0.297452 0.313831), ppl = 67.5174
    # iteration 17, lambda = (0.387511 0.297782 0.314707), ppl = 67.517
    # 45979 non-oov words, best lambda (0.386512 0.29803 0.315458)

fi