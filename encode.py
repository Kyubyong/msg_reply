'''
Generate data/cornell.txt that encodes cornell corpus.
It looks like this:
[sg_id] [text] [encoding]
0	You makin' any headway?	2017 5003 4939 1005 2151 2132 4576 1029 1064
0	She kissed me.	2016 4782 2033 1012 1064
200020	Where?	2073 1029 1064

'''

import re, os
import pickle
from hparams import hp
from pytorch_pretrained_bert import BertTokenizer
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import codecs


def refine(text):
    text = text.lower()
    text = re.sub("[^ A-Za-z\|]", "", text)
    return text

def get_utterances(line):
    text = re.search("\[(.+?)\]", line).group(1)
    text = re.sub("[',]", "", text)
    utts = text.split()
    if len(utts) < 2:
        print(line)
    return utts


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Load dictionaries
    phr2sg_id = pickle.load(open(hp.phr2sg_id, 'rb'))
    sg_id2phr = pickle.load(open(hp.sg_id2phr, 'rb'))

    # Load Cornell movie corpus
    convs = os.path.join(hp.corpus, "movie_conversations.txt")
    lines = os.path.join(hp.corpus, "movie_lines.txt")

    indices_li = [get_utterances(line) for line in codecs.open(convs, 'r', "utf-8").read().splitlines()]  # list of lists
    idx2utt = dict()
    for line in codecs.open(lines, 'r', "utf-8", errors="ignore").read().splitlines():
        cols = line.split("+++$+++")
        idx, utt = cols[0].strip(), cols[-1].strip()
        idx2utt[idx] = utt

    os.makedirs(os.path.dirname(hp.text), exist_ok=True)
    with open(hp.text, 'w') as fout:
        for i, indices in tqdm(enumerate(indices_li), total=len(indices_li)):
            if len(indices) < 2:
                print(indices)
            utts = [idx2utt[idx] for idx in indices]

            is_valid = True
            for utt in utts:
                if len(utt.strip()) < 1:
                    is_valid = False
                    break
            if not is_valid: continue

            for utt in utts:
                utt = utt.replace("\t", " ").replace("  ", " ")
                utt0 = sent_tokenize(utt)[0]
                utt0 = refine(utt0)
                sg_id = phr2sg_id.get(utt0, 0)

                tokens = tokenizer.tokenize(utt)[:512-1] # 512: max length of bert
                if len(tokens) == 0: continue
                tokens += ["|"]  # utterance delimiter
                ids = tokenizer.convert_tokens_to_ids(tokens)
                ids = " ".join(str(idx) for idx in ids)

                # save
                fout.write(f"{sg_id}\t{utt}\t{ids}\n")
            fout.write("\n")

