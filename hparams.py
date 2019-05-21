class Hparams:
    # construct_sg
    opus_en = "opensubtitles2018/OpenSubtitles.en-es.en"
    opus_es = "opensubtitles2018/OpenSubtitles.en-es.es"
    sg = "data/sg.en.es.json"

    # make_phr2sg_id
    min_cnt = 5 # a phrase whose count is 5 or more is included
    n_phrs = 10000 # number of phrases
    phr2sg_id = "data/phr2sg_id.pkl"
    sg_id2phr = "data/sg_id2phr.pkl"

    # encode
    corpus = "cornell movie-dialogs corpus"
    text = "data/cornell.txt"

    # prepro
    pkl_train = 'data/train.pkl'
    pkl_dev = 'data/dev.pkl'
    n_classes = 100
    phr2idx = "data/phr2idx.pkl"
    idx2phr = "data/idx2phr.pkl"

    # train
    batch_size = 32*8  # 8 GPUs
    lr = 2e-5
    logdir = 'log'
    vocab_size = 28996
    max_span = 128 # maximum token length for context
    n_train_steps = 10000

    # also test
    n_candidates = 5

hp = Hparams()