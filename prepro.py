'''
Make phr2idx, idx2phr, and {train|dev}.pkl

idx2phr:
{0: 'Yes!',
 1: 'Good answer.',
 2: 'What?',
 3: 'Good.',
 4: 'Of course I do.',
 5: "I don't know.",
 6: 'What for?',
 7: 'Oh!',
 8: 'Thank you.',
 9: 'Hello?',
 10: 'Right.',
 11: 'I know.',
 12: "What's wrong?",
 13: 'Really?',
 14: "Oh, I'm sorry.",
 15: 'Oh, yes!',
 16: 'Well...',
 17: 'Yes, sir?',
 18: 'Nothing.',
 19: 'Hi!',
 20: 'Huh!',
 21: 'Why not?',
 22: '10.',
 23: 'Who?',
 24: 'Stop it.',
 25: 'Shit!',
 26: 'What do you mean?',
 27: 'Aha.',
 28: 'Yes.',
 29: 'Come on!',
 30: 'Shut up!',
 31: 'What the hell are you talking about?',
 32: 'So.',
 33: 'Excuse me...',
 34: 'Which one?',
 35: 'What are you doing?',
 36: 'Where?',
 37: 'Oh, I see.',
 38: 'I beg you!',
 39: 'Me!',
 40: 'What happened?',
 41: 'Great!',
 42: 'Oh, no.',
 43: 'Jesus!',
 44: 'Maybe.',
 45: 'This is it.',
 46: 'Excuse me!',
 47: 'No.',
 48: 'I do.',
 49: 'Wait?',
 50: 'How?',
 51: 'No, thank you.',
 52: 'Forget it.',
 53: 'Just like me.',
 54: "I don't think so.",
 55: 'I...',
 56: 'We will.',
 57: 'Nonsense.',
 58: 'No, no',
 59: 'Oh, my God.',
 60: 'What is this?',
 61: 'Look!',
 62: "Can't I?",
 63: 'No, sir.',
 64: 'Here...',
 65: "I'm fine.",
 66: 'All right?',
 67: "I don't understand!",
 68: 'What do you want?',
 69: 'Wait a minute!',
 70: 'You!',
 71: 'How wonderful!',
 72: 'OK!',
 73: 'When was it?',
 74: 'All in order.',
 75: 'Did I?',
 76: 'I got it.',
 77: 'Nope.',
 78: 'Mmm?',
 79: 'Sir',
 80: 'Not a chance.',
 81: 'Who are you?',
 82: 'Good night...',
 83: 'Die!',
 84: 'What do you think?',
 85: 'Not exactly.',
 86: 'Where are you going?',
 87: 'Are you all right?',
 88: "I'm...",
 89: 'Like what?',
 90: 'I can imagine.',
 91: "Don't be afraid.",
 92: 'Huh?',
 93: 'Of course.',
 94: 'Bye!',
 95: 'Yeah.',
 96: 'Of course not!',
 97: 'I got it.',
 98: "No, it's not true.",
 99: 'What does that mean?'}

'''


from hparams import hp
import pickle, os
from tqdm import tqdm
from collections import Counter

def get_most_frequent_sgs(fin, n_classes):
    sg_ids = []
    for line in open(fin, 'r'):
        if len(line) > 1:
            sg_id = line.split("\t")[0]
            sg_id = int(sg_id)
            if sg_id != 0: # 0: non-sg
                sg_ids.append(sg_id)
    sg_id2cnt = Counter(sg_ids)
    sg_ids = [sg_id for sg_id, cnt in sg_id2cnt.most_common(n_classes)]
    idx2sg_id = {idx: sg_id for idx, sg_id in enumerate(sg_ids)}
    sg_id2idx = {sg_id: idx for idx, sg_id in enumerate(sg_ids)}
    return idx2sg_id, sg_id2idx

def prepro(fin, pkl_train, pkl_dev, n_classes, sg_id2idx):
    contexts_li = [[] for _ in range(n_classes)]

    entries = open(fin, 'r').read().split("\n\n")
    for entry in tqdm(entries):
        lines = entry.splitlines()
        for i, line in enumerate(lines):
            if i==0: continue
            cols = line.strip().split("\t")
            sg_id, sent, ids = cols
            sg_id = int(sg_id)
            if sg_id in sg_id2idx:
                idx = sg_id2idx[sg_id]
                ctx = [] # e.g. [ [3, 4, 5], [23, 9, 4, 5]  ]
                for l in lines[:i]:
                    ctx.append([int(id) for id in l.strip().split("\t")[-1].split()])
                contexts = contexts_li[idx]
                contexts.append(ctx)
    train, dev = [], []
    for contexts in contexts_li:
        if len(contexts) > 1:
            train.append(contexts[1:])
            dev.append(contexts[:1])
        else:
            train.append(contexts)
            dev.append([])


    pickle.dump(train, open(pkl_train, 'wb'))
    pickle.dump(dev, open(pkl_dev, 'wb'))
    print("done")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(hp.pkl_train), exist_ok=True)
    os.makedirs(os.path.dirname(hp.pkl_dev), exist_ok=True)

    idx2sg_id, sg_id2idx = get_most_frequent_sgs(hp.text, hp.n_classes)

    phr2sg_id = pickle.load(open(hp.phr2sg_id, 'rb'))
    sg_id2phr = pickle.load(open(hp.sg_id2phr, 'rb'))

    phr2idx = dict()
    for phr, sg_id in phr2sg_id.items():
        if sg_id in sg_id2idx:
            phr2idx[phr] = sg_id2idx[sg_id]

    idx2phr = dict()
    for idx, sg_id in idx2sg_id.items():
        if sg_id in sg_id2phr:
            idx2phr[idx] = sg_id2phr[sg_id]

    pickle.dump(phr2idx, open(hp.phr2idx, 'wb'))
    pickle.dump(idx2phr, open(hp.idx2phr, 'wb'))

    prepro(hp.text, hp.pkl_train, hp.pkl_dev, hp.n_classes, sg_id2idx)
    print("DONE")