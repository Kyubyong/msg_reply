# Smart Message Reply

Have you ever seen or used [Google Smart Reply](https://firebase.google.com/docs/ml-kit/generate-smart-replies)? It's a service that provides automatic reply suggestions for user messages. See below.

<img src="https://udelabs.com/wp-content/uploads/2019/04/post_thumbnail-3.png">

This is another name of the retrieval based chatbot, which is an important real life application, I think.
Let's build a simple message reply suggestion system.

Kyubyong Park <br>
Code-review by [Yj Choe](https://github.com/yjchoe)

## Basic Idea
* We need to set the list of suggestions to show. Naturally, frequency is considered first. But what about those phrases that are similar in meaning? For example, <i>hey</i> and <i>hi</i> are frequently used. Should they be treated independently? I don't think so. We have to group them. How?
* We make use of a parallel corpus. Both <i>hey</i> and <i>hi</i> are likely to be translated into the same language. Inspired by this, we construct English synonym groups that share the same translation.
* We need a large dialog corpus. It must consist of scenes, and each scene should consist of multiple turns.
* We use the Cornell Movie Dialogue Corpus. It's small, allowing for our purpose.
* If a turn starts with one of the pre-built suggestions, it is given a label.
* All turns function as a context.


## Requirements
* python>=3.6
* tqdm>=4.30.0
* pytorch>=1.0
* pytorch_pretrained_bert>=0.6.1
* nltk>=3.4

## Training
* STEP 0. Download OpenSubtitles 2018 Spanish-English Parallel data.
```
bash download.sh
```

* STEP 1. Construct synonym groups from the corpus.
```
python construct_sg.py
```
* STEP 2. Make phr2sg_id and sg_id2phr dictionaries.
```
python make_phr2sg_id.py
```
* STEP 3. Convert a monolingual English text to ids.
```
python encode.py
```
* STEP 4. Create training data and save them as pickle.
```
python prepro.py
```
* STEP 5. Train.
```
python train.py
```

## Test (Demo)

<img src="demo.gif">

* Download and extract the [pre-trained model](https://www.dropbox.com/s/fqomn5flbwlvndc/log.tar.gz?dl=0) and run the following command.
```
python test.py --ckpt log/9500_ACC0.1.pt
```

