# Create interpolated 4gram models using SRILM on PTB dataset
sh interpolate_ngrams.sh ptb 4
sh create_mix_ngram.sh ptb 4 0.635205 0.232781

# Create interpolated 4gram models using SRILM on SWBD dataset
sh interpolate_ngrams.sh swbd 4
sh create_mix_ngram.sh swbd 4 0.181104 0.734954

# Create interpolated 4gram models using SRILM on AMI dataset
sh interpolate_ngrams.sh ami 4
sh create_mix_ngram.sh ami 4 0.435916 0.498104



# Train LSTM-LM on PTB dataset
python main.py --train --dataset ptb --save trained/ptb-lstm --model lstm > trained/ptb-lstm.log &

# Train LSTM-GPNN-LM on PTB dataset
python main.py --train --dataset ptb --save trained/ptb-lstm-gpnn --model lstm-gpnn > trained/ptb-lstm-gpnn.log &

# Train LSTM-LM on SWBD dataset
python main.py --train --dataset swbd --save trained/swbd-lstm --model lstm > trained/swbd-lstm.log &

# Test PPL using only Ngram LM on SWBD dataset
python main.py --ppl --dataset swbd --only-ngram > results/ptb-4gram.ppl &

# Test PPL of LSTM-LM on SWBD dataset
python main.py --ppl --dataset swbd --load trained/swbd-lstm --model lstm > results/ptb-lstm-4gram.ppl &

# Test PPL of Ngram Interpolated LSTM-LM on SWBD dataset
python main.py --ppl --dataset swbd --load trained/swbd-lstm --model lstm --ngram > results/ptb-lstm-4gram.ppl &

# Nbest Rescoring using only Ngram LM on SWBD dataset
python main.py --rescore --dataset swbd --lmscale 12.0 --model lstm --load trained/swbd-lstm --save trained/swbd-4gram.hyp --only-ngram > trained/swbd-4gram-rescoring.log &

# Nbest Rescoring using LSTM-LM on SWBD dataset
python main.py --rescore --dataset swbd --lmscale 12.0 --model lstm --load trained/swbd-lstm --save trained/swbd-lstm.hyp > trained/swbd-lstm-rescoring.log &

# Nbest Rescoring using Ngram Interpolated LSTM-LM on SWBD dataset
python main.py --rescore --dataset swbd --lmscale 12.0 --model lstm --load trained/swbd-lstm --save trained/swbd-lstm-4gram.hyp --ngram > trained/swbd-lstm-4gram-rescoring.log &

# Train LSTM-GPNN-LM on SWBD dataset
python main.py --train --dataset swbd --save trained/swbd-lstm-gpnn --model lstm-gpnn > trained/swbd-lstm-gpnn.log &

# Train LSTM-LM on AMI dataset
python main.py --train --dataset ami --save trained/ami-lstm --model lstm > trained/ami-lstm.log &

# Train LSTM-GPNN-LM on AMI dataset
python main.py --train --dataset ami --save trained/ami-lstm-gpnn --model lstm-gpnn > trained/ami-lstm-gpnn.log &