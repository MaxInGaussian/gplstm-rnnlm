import argparse
import os, sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import gc
import data
import model

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

parser = argparse.ArgumentParser(description='Pytorch implementation of GPLSTM RNNLM training')
parser.add_argument('--cuda',       type=int,   default=0,      help='set cuda device id')
parser.add_argument('--multigpu',   action='store_true',        help='use multiple GPUs')

# Command Parameters
parser.add_argument('--train',      action='store_true',        help='train mode')
parser.add_argument('--finetune',   action='store_true',        help='finetune mode')
parser.add_argument('--cont',       action='store_true',        help='continue train/finetune from a checkpoint')
parser.add_argument('--ppl',        action='store_true',        help='write ppl mode')
parser.add_argument('--rescore',    action='store_true',        help='rescoring mode')
parser.add_argument('--interp',     action='store_true',        help='interpolation mode')

# Dataset Parameters
parser.add_argument('--data',       type=str,   required=False, help='choose the dataset: [ptb | swbd | callhm | ami]')
parser.add_argument('--trainfile',  type=str,   required=False, help='path of train data')
parser.add_argument('--validfile',  type=str,   required=False, help='path of valid data')
parser.add_argument('--testfile',   type=str,   required=False, help='path of test data')
parser.add_argument('--vocfile',    type=str,   required=False, help='path of vocabulary, format: id(<int>) word(<str>)')

# Model Parameters
parser.add_argument('--model',      type=str,   default='lstm', help='model type: [lstm | gplstm]')
parser.add_argument('--embsize',    type=int,   default=200,    help='embeeding layer size, default: 200')
parser.add_argument('--hiddensize', type=int,   default=-1,     help='hidden layer size, default: embsize')
parser.add_argument('--n_layers',   type=int,   default=1,      help='number of hidden layers, default: 1')
parser.add_argument('--dropout',    type=float, default=0.,     help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth',   type=float, default=0.,     help='dropout for RNN layers (0 = no dropout)')
parser.add_argument('--dropouti',   type=float, default=0.,     help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute',   type=float, default=0.,     help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--dropoutl',   type=float, default=0.,     help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--wdrop',      type=float, default=0.,     help='amount of weight dropout to apply to the RNN hidden matrix')
parser.add_argument('--tied',       action='store_false',       help='tie the word embedding and softmax weights')
parser.add_argument('--save',       type=str,                   help='path to save the trained model')
parser.add_argument('--load',       type=str,                   help='path to load the stored model')
parser.add_argument('--n_experts',  type=int,   default=1,      help='number of experts')

# Test PPL Parameters
parser.add_argument('--ngram',      action='store_true',        help='Ngram interpolated log prob')
parser.add_argument('--ngram_ppl',  type=str,   default=None,   help='load Ngram LM scored PPL')
parser.add_argument('--only_ngram', action='store_true',        help='only Ngram log prob')

# Rescoring Parameters
parser.add_argument('--lmscale',    type=float, default=12.0,   help='rescoring lmscale')
parser.add_argument('--ngramscale', type=float, default=.5,     help='interpolation weight of ngram in [0, 1]')
parser.add_argument('--pma_txt',    type=str,   default=None,   help='path of pma txt')
parser.add_argument('--pma_dir',    type=str,   default=None,   help='directory of pma files')
parser.add_argument('--pma_ppl',    type=str,   default=None,   help='load Ngram LM scored PMA PPL')

# Interpolation Parameters
parser.add_argument('--lambda1',    type=float, default=.5,     help='interpolation weight of ngram in [0, 1]')
parser.add_argument('--lambda2',    type=float, default=-1.,    help='pinterpolation weight of ngram in [0, 1]')
parser.add_argument('--ppl1',       type=str,   default=None,   help='load LM1 scored PMA PPL')
parser.add_argument('--ppl2',       type=str,   default=None,   help='load LM2 scored PMA PPL')
parser.add_argument('--ppl3',       type=str,   default=None,   help='load LM3 scored PMA PPL')

# Optimization Parameters
parser.add_argument('--lr',         type=float, default=18,    help='initial learning rate, default: 1.0')
parser.add_argument('--batchsize',  type=int,   default=30,     help='batch size, default: 30')
parser.add_argument('--smallbsz',   type=int,   default=10,     help='the small batch size for computation.\
    batchsize should be divisible by smallbsz. In our implementation, we compute gradients with\
    smallbsz multiple times, and accumulate the gradients until batchsize is reached.\
    An update step is then performed.')
parser.add_argument('--bptt',       type=int,   default=70,     help='sequence length, default: 70')
parser.add_argument('--max_seqlen', type=int,   default=40,     help='max sequence length')
parser.add_argument('--alpha',      type=float, default=2,      help='alpha L2 regularization on RNN activation')
parser.add_argument('--beta',       type=float, default=1,      help='beta slowness regularization applied on RNN activiation')
parser.add_argument('--wdecay',     type=float, default=1.2e-6, help='weight decay applied to all weights')
parser.add_argument('--clip',       type=float, default=0.25,   help='gradient clip, default: 0.25')
parser.add_argument('--seed',       type=int,   default=1,      help='random seed, default: 1')
parser.add_argument('--maxepoch',   type=int,   default=50,     help='maximum number of epoch for training, default: 50')
parser.add_argument('--asgd',       action='store_true',        help='switch to ASGD when SGD converges')
parser.add_argument('--log-int',    type=int,   default=200,    help='interval of log info, default: 200')

# GPLSTM Parameters
parser.add_argument('--uncertain',  type=str,   default=None,   help='uncertain type: [None | gp | bayes]')
parser.add_argument('--position',   type=int,   default=1,      help='uncertain position: [0-6]')

args = parser.parse_args()


basis_func_set = {'sigmoid', 'tanh', 'relu', 'cos', 'sin'}

if args.hiddensize < 0:
    args.hiddensize = args.embsize
if args.dropoutl < 0:
    args.dropoutl = args.dropouth
if args.smallbsz < 0:
    args.smallbsz = args.batchsize
if args.only_ngram:
    args.ngram = True
if args.data == 'ptb':
    args.trainfile = 'data/ptb.train.txt'
    args.validfile = 'data/ptb.valid.txt'
    args.testfile  = 'data/ptb.test.txt'
    args.vocfile   = 'data/ptb.voc.txt'
    args.ngram_ppl = 'data/ptb.ppl'
elif args.data == 'swbd':
    args.trainfile = 'data/swbd.train.txt'
    args.validfile = 'data/swbd.valid.txt'
    args.testfile  = 'data/swbd.test.txt'
    if args.model == 'lstm':
        args.vocfile   = 'swbd.voc.txt'
    else:
        args.vocfile   = 'data/swbd.voc.txt'
    args.ngram_ppl = 'data/swbd.ppl'
    if args.rescore or args.interp:
        args.pma_txt = 'data/pma/swbd_pma/swbd.txt'
        args.pma_ppl = 'data/pma/swbd_pma/swbd.ppl'
        args.pma_dir = 'data/pma/swbd_pma'
elif args.data == 'callhm':
    args.trainfile = 'data/swbd.train.txt'
    args.validfile = 'data/swbd.valid.txt'
    args.testfile  = 'data/swbd.test.txt'
    if(args.model == 'lstm'):
        args.vocfile   = 'swbd.voc.txt'
    else:
        args.vocfile   = 'data/swbd.voc.txt'
    args.ngram_ppl = 'data/swbd.ppl'
    if args.rescore or args.interp:
        args.pma_txt = 'data/pma/callhm_pma/callhm.txt'
        args.pma_ppl = 'data/pma/callhm_pma/callhm.ppl'
        args.pma_dir = 'data/pma/callhm_pma'
elif args.data == 'ami':
    args.trainfile = 'data/ami.ori.train.txt'
    args.validfile = 'data/ami.ori.valid.txt'
    args.testfile  = 'data/ami.ori.valid.txt'
    args.vocfile   = 'data/ami.voc.txt'
    args.ngram_ppl = 'data/ami.ppl'
    if args.rescore or args.interp:
        args.pma_txt = 'data/pma/ami_pma/ami.txt'
        args.pma_ppl = 'data/pma/ami_pma/ami.ppl'
        args.pma_dir = 'data/pma/ami_pma'

if not args.cont and not args.finetune:
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=['main.py', 'model.py', 'rnn.py'])

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if args.cuda < 0:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.set_device(args.cuda)
        torch.cuda.manual_seed_all(args.seed)

def logging(s, print_=True, log_=True):
    if print_:
        print(s)
        sys.stdout.flush()
    if log_:
        log_filename = ('finetune_'if args.finetune else '') +'log.txt'
        with open(os.path.join(args.save, log_filename), 'a+') as f_log:
            f_log.write(s + '\n')

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batchsize = 10
test_batchsize = 1
train_data = batchify(corpus.train, args.batchsize, args)
val_data = batchify(corpus.valid, eval_batchsize, args)
test_data = batchify(corpus.test, test_batchsize, args)

###############################################################################
# Build/Load the model
###############################################################################

vocsize = len(corpus.dictionary)
if args.cont:
    if args.train:
        model = torch.load(os.path.join(args.save, 'model.pt'))
    elif args.finetune:
        model = torch.load(os.path.join(args.save, 'finetune_model.pt'))
else:
    if args.train:
        model = model.RNNLM(args.model, vocsize, args.embsize, args.hiddensize, args.n_layers,
                            args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop,
                            args.tied, args.dropoutl, args.n_experts, args.uncertain, args.position)
    elif args.finetune:
        model = torch.load(os.path.join(args.save, 'model.pt'))
    
if args.cuda >= 0:
    if args.multigpu:
        model = nn.DataParallel(model, dim=1).cuda()
    else:
        model = model.cuda()
else:
    model = model

logging('Cmd: python {}'.format(' '.join(sys.argv)))
total_params = sum(x.data.nelement() for x in model.parameters())
logging('Args: {}'.format(args))
logging('Vocabulary size: %s'%(vocsize))
logging('Model total parameters: {}'.format(total_params))
print(model)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def train():
    assert args.batchsize % args.smallbsz == 0, 'batchsize must be divisible by smallbsz'

    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    vocsize = len(corpus.dictionary)
    hidden = [model.init_hidden(args.smallbsz) for _ in range(args.batchsize // args.smallbsz)]
    batch, i = 0, 0
    while i < train_data.size(0)-2:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + args.max_seqlen)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        optimizer.zero_grad()

        start, end, s_id = 0, args.smallbsz, 0
        while start < args.batchsize:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden[s_id] = repackage_hidden(hidden[s_id])

            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = model(cur_data, hidden[s_id], return_h=True)
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss
            # KL-divergence Regularization for GPLSTM
            if(args.uncertain == 'gp' or args.uncertain == 'bayes'):
                loss = loss + model.rnns[0].kl_divergence()/train_data.size(0)*args.batchsize
            
            if(args.alpha > 0):
                # Activiation Regularization
                loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean()
                                  for dropped_rnn_h in dropped_rnn_hs[-1:])
                
            if(args.beta > 0):
                # Temporal Activation Regularization (slowness)
                loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
                
            loss *= args.smallbsz / args.batchsize
            total_loss += raw_loss.data * args.smallbsz / args.batchsize
            loss.backward()

            s_id += 1
            start = end
            end = start + args.smallbsz

            gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_int == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_int
            elapsed = time.time() - start_time
            logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data)//args.bptt, optimizer.param_groups[0]['lr'],
                elapsed*1000/args.log_int, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

def evaluate(data_source, batchsize=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    samplesize = 10
    total_loss = 0
    vocsize = len(corpus.dictionary)
    hidden = model.init_hidden(batchsize)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args)
            targets = targets.view(-1)
            log_prob, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

            total_loss += loss * len(data)

    return total_loss.item() / len(data_source)

if args.rescore:
    if(not args.only_ngram):
        batchsize = args.batchsize
        PmaData  = txt2dataset(args.pma_txt, voc)
        pmadataloader  = Data.DataLoader(PmaData , batchsize=batchsize,
                                         shuffle=False, num_workers=0,
                                         collate_fn=collate_fn, drop_last=True)

def rescore(pma_dir, sent2logprob):
    if(args.dataset == 'swbd'):
        rec_ids, best_sents = [], []
        pma_files = get_all_pma_files_in_directory(pma_dir)
        for pma_file in pma_files:
            rec_id = pma_file.split('/')[-1][:-4]
            rec_id = 'swXXXX-'+'-'.join(rec_id.split('-')[2:])
            rec_ids.append(rec_id)
            best_sents.append(nbest_rescoring_on_pma_file(pma_file, sent2logprob, args.lmscale))
            print(rec_id, best_sents[-1])
        write_hyp = ''
        for rec_id, sent in zip(rec_ids, best_sents):
            write_hyp += "\"./"+rec_id+".rec\" "+sent+' \n'
        with open (args.save, 'wt') as f:
            f.write(write_hyp)
    elif(args.dataset == 'callhm'):
        rec_ids, best_sents = [], []
        pma_files = get_all_pma_files_in_directory(pma_dir)
        for pma_file in pma_files:
            rec_id = pma_file.split('/')[-1][:-4]
            rec_id = 'enXXXX-'+'-'.join(rec_id.split('-')[2:])
            rec_ids.append(rec_id)
            best_sents.append(nbest_rescoring_on_pma_file(pma_file, sent2logprob, args.lmscale))
            print(rec_id, best_sents[-1])
        write_hyp = ''
        for rec_id, sent in zip(rec_ids, best_sents):
            write_hyp += "\"./"+rec_id+".rec\" "+sent+' \n'
        with open (args.save, 'wt') as f:
            f.write(write_hyp)
    elif(args.dataset == 'ami'):
        best_sents = []
        dev_pma_files = get_all_pma_files_in_directory(pma_dir, 'data/pma/ami_pma/mdm_rt05dev.pmalist')
        for pma_file in dev_pma_files:
            best_sents.append(nbest_rescoring_on_pma_file(pma_file, sent2logprob, args.lmscale))
            print(pma_file, best_sents[-1])
        with open (args.save+'.dev', 'wt') as f:
            f.write('\n'.join(best_sents)+'\n\r')
        best_sents = []
        eval_pma_files = get_all_pma_files_in_directory(pma_dir, 'data/pma/ami_pma/mdm_rt05eval.pmalist')
        for pma_file in eval_pma_files:
            best_sents.append(nbest_rescoring_on_pma_file(pma_file, sent2logprob, args.lmscale))
            print(pma_file, best_sents[-1])
        with open (args.save+'.eval', 'wt') as f:
            f.write('\n'.join(best_sents)+'\n\r')

def calppl(dataloader, ngram_ppl):
    model.eval()
    loss, nword = 0., 0
    sents, nglm_sent2logprob = read_ppl(ngram_ppl)
    rnnlm_sent2logprob = nglm_sent2logprob.copy()
    with torch.no_grad():
        print ('{:<10} {:<12} {:<12} {:<10}'.format('Word', 'Prob', 'Log Prob', 'Test PPL'))
        for chunk, (input, target, sent_lens) in enumerate(dataloader):
            target_packed = pack_padded_sequence(target, sent_lens)[0]
            if(not args.only_ngram):
                output_nn = model(input, target, sent_lens)
                output = output_nn.view(-1, vocsize)
                output = torch.nn.functional.softmax(output, dim=1)
            hor_end_idx = (target.cpu().numpy()!=0).argmin(axis=1)
            for i, end_idx in enumerate(hor_end_idx):
                if(end_idx == 0):
                    hor_end_idx[i] = batchsize
                else:
                    break
            hor_end_idx = [0]+np.cumsum(hor_end_idx).tolist()
            for bid in range(batchsize):
                sent_ids = target[:sent_lens[bid]-1, bid].cpu().numpy().tolist()
                sent = tuple(map(voc.id2word, sent_ids))
                if(sent not in nglm_sent2logprob):
                    for ngram_sent in nglm_sent2logprob.keys():
                        found_possible = False
                        for word in sent:
                            if(word not in ngram_sent and word != UNK):
                                break
                        else:
                            if(len(ngram_sent) == len(sent)):
                                found_possible = True
                        for word in ngram_sent:
                            if(word not in sent and word != UNK):
                                break
                        else:
                            if(len(ngram_sent) == len(sent)):
                                found_possible = True
                        if(found_possible):
                            print(sent)
                            print(ngram_sent)
                            for ngram_word, word in zip(ngram_sent, sent):
                                if(ngram_word == word):
                                    continue
                                if(ngram_word != word and (ngram_word == UNK or word == UNK)):
                                    continue
                                break
                            else:
                                # found corresponding sent in ngram model
                                ngram_logprobs = nglm_sent2logprob[ngram_sent]
                                sent = ngram_sent
                                break
                else:
                    ngram_logprobs = nglm_sent2logprob[sent]
                for rid in range(sent_lens[bid].item()):
                    wid = target_packed[hor_end_idx[rid]+bid].item()
                    word = voc.id2word(wid)
                    if(not args.only_ngram):
                        rnnlm_prob = output[hor_end_idx[rid]+bid, wid].item()
                    if(args.ngram):
                        if(rid < len(sent) and sent[rid] == UNK):
                            continue
                        # Conditional probability p(w_t|w_{t-1}...w{t-n+1})
                        ngram_prob = math.pow(10, ngram_logprobs[rid])
                        if(args.only_ngram):
                            prob = ngram_prob
                        else:
                            prob = rnnlm_prob*(1-args.ngramscale)+ngram_prob*args.ngramscale
                    else:
                        prob = rnnlm_prob
                    nword += 1
                    log_prob = math.log(prob)
                    if(not args.only_ngram):
                        rnnlm_sent2logprob[sent][rid] = math.log10(prob)
                    loss += log_prob
                    print ('{:<10} {:.10f} {:.10f}, {:.10f}'.format(
                        voc.id2word(wid), prob, log_prob, math.exp(-loss/nword)))
    return -loss/nword, rnnlm_sent2logprob


# At any point you can hit Ctrl + C to break out of training early.
try:
        
    if args.train:
        if args.cont:
            optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
            if 't0' in optimizer_state['param_groups'][0]:
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
            optimizer.load_state_dict(optimizer_state)
            stored_loss = evaluate(val_data)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
            stored_loss = np.Infinity
        
        best_val_loss = []
        val_loss, count_div, tol_div = np.Infinity, 0, 1e-3
        
        for epoch in range(1, args.maxepoch+1):
            epoch_start_time = time.time()
            train()
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()

                val_loss2 = evaluate(val_data)
                logging('-' * 89)
                logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                   val_loss2, math.exp(val_loss2)))
                logging('-' * 89)

                if val_loss2 < stored_loss:
                    save_checkpoint(model, optimizer, args.save)
                    logging('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

            else:
                val_loss = evaluate(val_data, eval_batchsize)
                logging('-' * 89)
                logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                   val_loss, math.exp(val_loss)))
                logging('-' * 89)

                if val_loss < stored_loss:
                    save_checkpoint(model, optimizer, args.save)
                    logging('Saving Normal!')
                    stored_loss = val_loss
                
                if (len(best_val_loss) > 3 and val_loss > min(best_val_loss[:-3])):
                    if optimizer.param_groups[0]['lr']*3**3 < args.lr:
                        if 't0' not in optimizer.param_groups[0] and args.asgd:
                            logging('Switching!')
                            optimizer = torch.optim.ASGD(model.parameters(),
                                                         lr=args.lr, t0=0, lambd=0.,
                                                         weight_decay=args.wdecay)
                        else:
                            logging('Done!')
                            break
                    else:
                        logging('Lower Learning Rate!')
                        optimizer.param_groups[0]['lr'] /= 3.
                    
                best_val_loss.append(val_loss)
                
    if args.finetune:
        if args.cont:
            optimizer_state = torch.load(os.path.join(args.save, 'finetune_optimizer.pt'))
            optimizer.load_state_dict(optimizer_state)
        else:
            optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr*1.25, t0=0, lambd=0., weight_decay=args.wdecay)
        
        best_val_loss = []
        stored_loss = evaluate(val_data)
        val_loss, count_div, tol_div = np.Infinity, 0, 1e-3
        
        for epoch in range(1, args.maxepoch+1):
            epoch_start_time = time.time()
            train()
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()

                val_loss2 = evaluate(val_data)
                logging('-' * 89)
                logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                   val_loss2, math.exp(val_loss2)))
                logging('-' * 89)

                if val_loss2 < stored_loss:
                    save_checkpoint(model, optimizer, args.save, finetune=True)
                    logging('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

            if (len(best_val_loss) > 5 and val_loss2 > min(best_val_loss[:-5])):
                logging('Done!')
                break
            best_val_loss.append(val_loss2)

    elif args.ppl:
        if(not args.only_ngram):
            with open (args.load, 'rb') as f:
                model = torch.load(f, map_location='cpu')
                if torch.cuda.is_available():
                    model.to(device)
                model.batchsize = args.batchsize
            print(model)
        test_loss, rnnlm_sent2logprob = calppl(testdataloader, args.ngram_ppl)
        sys.stdout.write('avglog loss {:5.2f}, PPL {:8.2f}\n'.format(
            test_loss, math.exp(test_loss)))
        sys.stdout.flush()
        if(not args.only_ngram and not args.ngram):
            write_ppl(rnnlm_sent2logprob, args.ngram_ppl,
                      args.load.split('.')[0]+('-4gram' if args.ngram else '')+'.ppl')

    elif args.rescore:
        if(not args.only_ngram):
            pma_ppl_path = args.pma_txt.split('.')[0]+'-'\
                +args.model+('-4gram' if args.ngram else '')+'.ppl'
            if(os.path.exists(pma_ppl_path)):
                sents, sent2logprob = read_ppl(pma_ppl_path)
            else:
                with open (args.load, 'rb') as f:
                    model = torch.load(f, map_location='cpu')
                    if torch.cuda.is_available():
                        model.to(device)
                    model.batchsize = args.batchsize
                print(model)
                test_loss, sent2logprob = calppl(pmadataloader, args.pma_ppl)
                write_ppl(sent2logprob, args.pma_ppl, pma_ppl_path)
        else:
            sents, sent2logprob = read_ppl(args.pma_ppl)
        rescore(args.pma_dir, sent2logprob)

    elif args.interp:
        if(args.lambda2 < 0):
            _, sent2logprob = read_ppl(args.ppl1)
            _, sent2logprob2 = read_ppl(args.ppl2)
            for sent, logprobs in sent2logprob.items():
                sent2logprob[sent] = np.log10(args.lambda1*(10**np.asarray(logprobs))+
                    (1-args.lambda1)*(10**np.asarray(sent2logprob2[sent]))).tolist()
        else:
            _, sent2logprob = read_ppl(args.ppl1)
            _, sent2logprob2 = read_ppl(args.ppl2)
            _, sent2logprob3 = read_ppl(args.ppl3)
            for sent, logprobs in sent2logprob.items():
                sent2logprob[sent] = np.log10(args.lambda1*(10**np.asarray(logprobs))+
                    args.lambda2*(10**np.asarray(sent2logprob2[sent]))+
                    (1-args.lambda1-args.lambda2)*(10**np.asarray(sent2logprob3[sent]))).tolist()
        rescore(args.pma_dir, sent2logprob)

except KeyboardInterrupt:
    logging('-' * 89)
    logging('Exiting from training early')

# Load the best saved model.
model = torch.load(os.path.join(args.save, 'model.pt'))
if args.cuda >= 0:
    if args.multigpu:
        model = nn.DataParallel(model, dim=1).cuda()
    else:
        model = model.cuda()
else:
    model = model

# Run on test data.
test_loss = evaluate(test_data, test_batchsize)
logging('=' * 89)
logging('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logging('=' * 89)


