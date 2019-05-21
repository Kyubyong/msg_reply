from hparams import hp
import torch
from model import Net
from pytorch_pretrained_bert import BertTokenizer
from collections import OrderedDict
from colorama import Fore, Style
import pickle, re

import argparse

def prepare_inputs(context, tokenizer):
    '''context
    context: I love you. [SEP] Sorry, I hate you.
    '''
    tokens = tokenizer.tokenize(context)
    tokens = tokenizer.convert_tokens_to_ids(tokens)[-hp.max_span+2:]
    tokens = [101] + tokens + [102]
    # print(f"{Fore.LIGHTBLACK_EX}context:{tokenizer.convert_ids_to_tokens(tokens)}{Style.RESET_ALL}")
    tokens = torch.LongTensor(tokens)
    tokens = tokens.unsqueeze(0) # (1, T)
    tokens = tokens.to("cuda")
    return tokens

def suggest(context, tokenizer, model, idx2phr):
    x = prepare_inputs(context, tokenizer)
    model.eval()
    with torch.no_grad():
        _, y_hat, y_hat_prob = model(x)
        y_hat = y_hat.cpu().numpy().flatten()  # (3)
        y_hat_prob = y_hat_prob.cpu().numpy().flatten()  # (3)
        y_hat_prob = [round(each, 2) for each in y_hat_prob]
        preds = [idx2phr.get(h, "None") for h in y_hat]
        preds = " | ".join(preds)
        print(f"{Fore.RED}{preds}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{y_hat_prob}{Style.RESET_ALL}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint file path")
    args = parser.parse_args()


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    print("Wait... loading model")
    ckpt = args.ckpt

    model = Net(hp.n_classes)
    model = model.cuda()
    ckpt = torch.load(ckpt)
    # model.load_state_dict(ckpt)

    # ckpt = OrderedDict([(k.replace("module.", "").replace("LayerNorm.weight", "LayerNorm.gamma").replace("LayerNorm.bias", "LayerNorm.beta"), v) for k, v in ckpt.items()])
    ckpt = OrderedDict([(k.replace("module.", ""), v) for k, v in ckpt.items()])
    model.load_state_dict(ckpt)
    print("Model loaded.")

    print("# loading dictionaries ..")
    idx2phr = pickle.load(open(hp.idx2phr, 'rb'))

    context = ""
    print("Let's start a conversation. If you want to start a new one, please press Enter.")
    while True:
        line = input("A:")
        if line == "":
            context = ""
            print("NEW CONVERSATION---")
            continue
        else:
            context += line + " | "

        suggest(context, tokenizer, model, idx2phr)

        line = input("B:")
        if line == "":
            context = ""
            print("NEW CONVERSATION---")
            continue
        else:
            context += line + " | "

        suggest(context, tokenizer, model, idx2phr)

