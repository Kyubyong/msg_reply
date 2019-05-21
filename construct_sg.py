'''
Construct synonym groups looking like this:

[
"0": {
        "_translation": "Te pondré más.",
        "phrases": [
            [
                "I'll give you more.",
                1
            ],
            [
                "You'll have to have some more.",
                1
            ],
...
]
'''

import json
from collections import Counter
from operator import itemgetter
from hparams import hp
import os
from tqdm import tqdm

def normalize(text):
    text = text.strip(" -\n")
    return text

if __name__ == "__main__":
    # Group phrases
    es2ens = dict()
    en_lines = open(hp.opus_en, 'r').read().splitlines()
    es_lines = open(hp.opus_es, 'r').read().splitlines()
    for en, es in tqdm(zip(en_lines, es_lines), total=len(en_lines)):
        en = normalize(en)
        es = normalize(es)
        if len(es) <= 1: continue
        if es not in es2ens: es2ens[es] = []
        es2ens[es].append(en)
    print(f"Grouped all synonymous phrases: {len(es2ens)}")

    # Sort
    data = dict()
    i = 0
    for es, ens in es2ens.items():
        en2cnt = Counter(ens)
        phrases = sorted(en2cnt.items(), key=itemgetter(1), reverse=True)
        if len(phrases) > 1:
            val = dict()
            val["_translation"] = es
            val["phrases"] = phrases
            data[i] = val
            i += 1
    print(f"Sorted all synonymous groups by frequency: {len(data)}")

    # Write
    os.makedirs(os.path.dirname(hp.sg), exist_ok=True)
    with open(hp.sg, 'w') as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4, separators=(',', ': '), sort_keys=True)
