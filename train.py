import torch
import torch.nn as nn
import torch.optim as optim
from data_load import get_batch
from hparams import  hp
from model import Net
from tqdm import tqdm
import os
import random
from pytorch_pretrained_bert import BertTokenizer
import pickle

def train_and_eval(model, optimizer, criterion, ids2tokens, idx2phr):
    model.train()
    for step in tqdm(range(hp.n_train_steps+1)):
        x, y = get_batch(hp.max_span, hp.batch_size, hp.n_classes, True)
        x = x.cuda()
        y = y.cuda()

        optimizer.zero_grad()

        logits, y_hat, _ = model(x) # logits: (N, classes), y_hat: (N,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        # evaluation
        if step and step%500==0: # monitoring
            eval(model, f'{hp.logdir}/{step}', ids2tokens, idx2phr)
            print(f"step: {step}, loss: {loss.item()}")
            model.train()

def eval(model, f, ids2tokens, idx2phr):
    model.eval()

    Y, Y_hat = [], []
    with torch.no_grad():
        x, y  = get_batch(hp.max_span, hp.batch_size, hp.n_classes, False)
        x = x.cuda()

        _, y_hat, _ = model(x)  # y_hat: (N, n_candidates)

        x = x.cpu().numpy().tolist()
        y = y.cpu().numpy().tolist()
        y_hat = y_hat.cpu().numpy().tolist()

        Y.extend(y)
        Y_hat.extend(y_hat)

        # monitoring
        pointer = random.randint(0, len(x)-1)
        xx, yy, yy_hat = x[pointer], y[pointer], y_hat[pointer] # one sample

        tokens = ids2tokens(xx) # this is a function.
        ctx = " ".join(tokens).replace(" ##", "").split("[PAD]")[0] # bert detokenization
        gt = idx2phr[yy] # this is a dict.
        ht = " | ".join(idx2phr[each] for each in yy_hat)

        print(f"context: {ctx}")
        print(f"ground truth: {gt}")
        print(f"predictions: {ht}")

    # calc acc.
    n_samples = len(Y)
    n_correct = 0
    for y, y_hat in zip(Y, Y_hat):
        if y in y_hat:
            n_correct += 1
    acc = n_correct / n_samples
    print(f"acc@{hp.n_candidates}: %.2f"%acc)

    acc = str(round(acc, 2))

    torch.save(model.state_dict(), f"{f}_ACC{acc}.pt")


if __name__=="__main__":
    os.makedirs(hp.logdir, exist_ok=True)

    print("==== Load tokenizer")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    ids2tokens = tokenizer.convert_ids_to_tokens

    print("==== Load dictionaries")
    idx2phr = pickle.load(open(hp.idx2phr, 'rb'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("==== Building model")
    model = Net(hp.n_classes)
    model = model.to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss()

    train_and_eval(model, optimizer, criterion, ids2tokens, idx2phr)


