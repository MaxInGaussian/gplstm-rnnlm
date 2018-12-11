import os, sys, shutil
import torch
from torch.autograd import Variable


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, tuple) or isinstance(h, list):
        return tuple(repackage_hidden(v) for v in h)
    else:
        return h.detach()

def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    print(data.size())
    if args.cuda >= 0:
        data = data.cuda()
    return data

def get_batch(source, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len])
    # target = Variable(source[i+1:i+1+seq_len].view(-1))
    target = Variable(source[i+1:i+1+seq_len])
    return data, target

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def save_checkpoint(model, optimizer, path, finetune=False):
    if finetune:
        torch.save(model, os.path.join(path, 'finetune_model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model, os.path.join(path, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
        

def collate_fn(batch):
    sent_lens = torch.LongTensor(list(map(len, batch)))
    maxlen = sent_lens.max()
    batchsize = len(batch)
    sent_batch = sent_lens.new_zeros(batchsize, maxlen)
    for idx, (sent, sent_len) in enumerate (zip(batch, sent_lens)):
        sent_batch[idx, :sent_len] = torch.LongTensor(sent)
    
    sent_lens, perm_idx = sent_lens.sort(0, descending=True)
    sent_batch = sent_batch[perm_idx]
    sent_batch = sent_batch.t().contiguous()

    inputs = sent_batch[0:maxlen-1]
    targets = sent_batch[1:maxlen]
    sent_lens.sub_(1)

    return inputs.to(device), targets.to(device), sent_lens.to(device)

def nbest_rescoring_on_pma_file(pma_file, sent2logprob, lmscale):
    pma_lines = open(pma_file, 'rt').readlines()
    scores, lm_scores, sents = [], [], []
    for pma_line in pma_lines:
        pma_infos = pma_line.strip().split()
        acoustic_score, _lm_score = list(map(float, pma_infos[:2]))
        rec_len, pma_words = int(pma_infos[2]), pma_infos[3:]
        sent = pma_words[:]
        if(sent[0] == SOS and sent[-1] == EOS):
            sent = sent[1:-1]
        sent = tuple(sent)
        if(sent not in sent2logprob):
            sent = list(sent)
            for rid, word in enumerate(sent):
                word = word.replace('\'', '\\\'')
                sent[rid] = word.replace('\\\\', '\\')
            sent = tuple(sent)
            for scored_sent in sent2logprob.keys():
                for word in scored_sent:
                    if(word not in sent and word != UNK):
                        break
                else:
                    valid_sent = []
                    for rid, word in enumerate(scored_sent):
                        if(word in sent or word == UNK):
                            valid_sent.append(word)
                    if(len(valid_sent) == len(sent)):
                        for valid_word, word in zip(valid_sent, sent):
                            if(valid_word == word):
                                continue
                            if(valid_word != word and valid_word == UNK):
                                continue
                            break
                        else:
                            # found corresponding sent in ngram model
                            sent = scored_sent[:]
                            break
        if(sent not in sent2logprob):
            # unk words
            print('UNK:', sent)
            sent = (UNK,)
        lm_score = np.sum(np.array(sent2logprob[sent]))
        lm_scores.append(lm_score)
        scores.append(acoustic_score+lmscale*lm_score)
        if(pma_words[0] == SOS and pma_words[-1] == EOS):
            pma_words = pma_words[1:-1]
        for w_i, word in enumerate(pma_words):
            pma_words[w_i] = word.replace("\\'", "'")
            if(word[0] == '\''):
                pma_words[w_i] = '\\' + word
        sents.append(' '.join(pma_words))
    argmax = np.argmax(scores)
    return sents[argmax]

def read_ppl(path):
    sents = []
    sent2logprob = {}
    with open(path, 'rt') as f:
        lines = f.readlines()
        is_sent = True
        for line_no, line in enumerate(lines):
            if('file' in line[:4] and ':' in line):
                break
            if(line.split()):
                if(is_sent):
                    sent = tuple(line.strip().split())
                    sents.append(sent)
                    sent2logprob[sent] = []
                else:
                    if('p(' in line):
                        if (UNK in line):
                            logprob = float(-10.)
                        else:
                            logprob = float(line.strip().split(' [ ')[-1].replace(' ]', ''))
                        sent2logprob[sent].append(logprob)
                is_sent = False
            else:
                is_sent = True
    return sents, sent2logprob


def write_ppl(sent2logprob, sample_ppl_path, save_path):
    with open(sample_ppl_path, 'rt') as fr:
        lines = fr.readlines()
        edited_lines = lines[:]
        with open(save_path, 'wt') as fw:
            is_sent, sent_count = True, 0
            for line_no, line in enumerate(edited_lines):
                if('file' in line[:4] and ':' in line):
                    break
                if(line.split()):
                    if(is_sent):
                        sent = tuple(line.strip().split())
                        if(sent not in sent2logprob):
                            # unk words
                            print('UNK:', sent)
                            sent = (UNK,)
                    else:
                        if('p(' in line and 'gram]' in line):
                            edited_lines[line_no] = line[:line.index(" [ ")+3]\
                                +'%.6f'%(sent2logprob[sent][sent_count])\
                                +line[line.index(" ]"):]
                            sent_count += 1
                    is_sent = False
                else:
                    is_sent, sent_count = True, 0
            fw.writelines(edited_lines)

def get_all_pma_files_in_directory(pma_dir, pma_list_path=None):
    pma_files = []
    for root, dirs, files in os.walk(pma_dir):
        for file in files:
            if file.endswith(".pma") and 'checkpoint' not in file:
                pma_files.append(os.path.join(root, file))
    if(pma_list_path is not None):
        pma_files_in_list = []
        with open(pma_list_path, 'rt') as f:
            pma_list = f.readlines()
            for line in pma_list:
                pma_name = '-'.join(line.strip().split('-')[1:])
                for pma_file in pma_files:
                    if(pma_name in pma_file):
                        pma_files_in_list.append(pma_file)
                        break
            pma_files = pma_files_in_list[:]
    return pma_files


def convert_pma_files_to_test_file(pma_dir, test_file_path):
    pma_txts = ''
    pma_files = get_all_pma_files_in_directory(pma_dir)
    for pma_file in pma_files:
        with open(pma_file, 'rt') as f:
            pma = f.readlines()
            for line in pma:
                pma_infos = line.strip().split()
                acoustic_score, lm_score = list(map(float, pma_infos[:2]))
                rec_len, pma_txt = int(pma_infos[2]), ' '.join(pma_infos[4:-1])
                pma_txt = pma_txt.replace("'", "\\'")
                pma_txt = pma_txt.replace("\\\\", "\\")
                pma_txts += pma_txt+'\n'
    with open(test_file_path, 'wt') as f:
        f.write(pma_txts)

def unstandardize_vocabulary(voc_path):
    with open(voc_path, 'rt') as fr:
        vocabs = set()
        words = fr.read().strip().split('\n')
        for word in words:
            if('\'' in word):
                vocabs.add(word.replace('\\\'', '\''))
            else:
                vocabs.add(word)
    with open(voc_path, 'wt') as fw:
        fw.write('\n'.join(list(sorted(list(vocabs)))))
    return vocabs

def standardize_vocabulary(voc_path):
    with open(voc_path, 'rt') as fr:
        vocabs = set()
        words = fr.read().strip().split('\n')
        for word in words:
            if('\'' in word):
                ind = word.index('\'')
                if(word[ind-1] != '\\'):
                    vocabs.add(word.replace('\'', '\\\''))
                else:
                    vocabs.add(word)
            else:
                vocabs.add(word)
    with open(voc_path, 'wt') as fw:
        fw.write('\n'.join(list(sorted(list(vocabs)))))
    return vocabs


def update_text_file(voc, text_path, no_change=True):
    unks = {}
    with open(text_path, 'rt') as fr:
        lines = fr.readlines()
        new_lines = []
        for i, line in enumerate(lines):
            if(len(line.strip().split()) == 0):
                continue
            content = []
            for word in line.strip().split():
                if('\'' in word):
                    ind = word.index('\'')
                    if(word[ind-1] != '\\'):
                        word = word.replace('\'', '\\\'')
                if word in voc:
                    content.append(word)
                else:
                    unks[word] = 1 if word not in unks else unks[word]+1
                    content.append(UNK)
            new_lines.append(' '.join(content))
    if(not no_change):
        with open(text_path, 'wt') as fw:
            fw.write('\n'.join(new_lines))
    return unks

def update_text_file_for_ngram(text_path, no_change=True):
    with open(text_path, 'rt') as fr:
        lines = fr.readlines()
        new_lines = []
        for i, line in enumerate(lines):
            if(len(line.strip().split()) == 0):
                continue
            content = []
            for word in line.strip().split():
                if('\\\'' in word and '\\' != word[0]):
                    word = word.replace('\\\'', '\'')
                if(word == 'ok'):
                    word = 'okay'
                content.append(word)
            new_lines.append(' '.join(content))
    if(not no_change):
        with open(text_path, 'wt') as fw:
            fw.write('\n'.join(new_lines))

def update_standardize_vocabulary(unks, voc_path, thresold=100):
    vocabs = set()
    for word, freq in unks.items():
        if(freq >= thresold):
            vocabs.add(word)
    with open(voc_path, 'rt') as fr:
        words = fr.read().strip().split('\n')
        for word in words:
            vocabs.add(word)
    with open(voc_path, 'wt') as fw:
        fw.write('\n'.join(list(sorted(list(vocabs)))))
    return vocabs