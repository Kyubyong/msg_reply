'''
Make two dictionaries: phr2sg_id and sg_id2phr

phr2sg_id["nice work']==6152
phr2sg_id["nicely done']==6152
phr2sg_id["nice going']==6152
sg_id2phr[6152]=="Well done."

'''


import json, os
import operator
import pickle
from hparams import hp
import re
from tqdm import tqdm

def refine(text):
    text = text.lower()
    text = re.sub("[^ A-Za-z]", "", text)
    return text

if __name__ == "__main__":
    print("Determine the most frequent Synonym Groups")
    data = json.load(open(hp.sg))
    sg_id2cnt = dict()
    for sg_id, sg in tqdm(data.items()):
        sg_id = int(sg_id)
        phrs = sg["phrases"] # [['i am mormon', 1], ["i'm a mormon", 1]]
        sg_cnt = 0 # total cnt
        for phr, cnt in phrs:
            if cnt >= hp.min_cnt:
                sg_cnt += cnt

        sg_id2cnt[sg_id] = sg_cnt

    sg_id_cnt = sorted(sg_id2cnt.items(), key=operator.itemgetter(1), reverse=True)
    sg_ids = [sg_id for sg_id, _ in sg_id_cnt][:hp.n_phrs]

    print("Determine the group of phrases")
    sg_id2phr = dict()
    phr2sg_id, phr2cnt = dict(), dict()
    for sg_id in tqdm(sg_ids):
        sg = data[str(sg_id)]
        phrs = sg["phrases"]  # [['i am mormon', 1], ["i'm a mormon", 1]]

        sg_id2phr[sg_id] = phrs[0][0]
        for phr, cnt in phrs:
            if cnt >= hp.min_cnt:
                phr = refine(phr)
                if phr in phr2cnt and cnt > phr2cnt[phr]: # overwrite
                    phr2cnt[phr] = cnt
                    phr2sg_id[phr] = sg_id
                else:
                    phr2cnt[phr] = cnt
                    phr2sg_id[phr] = sg_id

    print("save")
    os.makedirs(os.path.dirname(hp.phr2sg_id), exist_ok=True)
    os.makedirs(os.path.dirname(hp.sg_id2phr), exist_ok=True)
    pickle.dump(phr2sg_id, open(hp.phr2sg_id, 'wb'))
    pickle.dump(sg_id2phr, open(hp.sg_id2phr, 'wb'))