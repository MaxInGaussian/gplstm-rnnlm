# Calculating the Test Data Perplexity
ngram -unk\
    -order ${2}\
    -ppl 'data/'${1}'.test.ngram.txt'\
    -lm 'data/'${3}'.arpa'\
    -debug 2 > 'data/'${1}'.ppl'

# Calculating the PMA Data Perplexity
ngram -unk\
    -order ${2}\
    -ppl 'data/pma/'${1}'_pma/'${1}'.ngram.txt'\
    -lm 'data/'${3}'.arpa'\
    -debug 2 > 'data/pma/'${1}'_pma/'${1}'.ppl'