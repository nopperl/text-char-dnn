Character-level Text Classification 
===
Implementation of character-level deep neural networks for text classification. Three models ([CNN](https://arxiv.org/abs/1509.01626), [VDCNN](https://arxiv.org/abs/1606.01781) and [GRU](https://arxiv.org/abs/1406.1078)) are evaluated on four binary text classification datasets ([Blog Authorship Corpus](http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm), [PAN13 and PAN14](https://pan.webis.de/data.html) and [Enron Email Dataset](https://www.cs.cmu.edu/~./enron/)). Results:

| | Blogs | PAN13 | PAN14 | Enron |
| --- | --- | --- | --- | --- |
| CNN | 65% | 55% | __69%__ | 57% |
| VDCNN | __66%__ | __74%__  | 67% | __64%__ |
| GRU | 62% | 60% | 63% | 62%|

Overall, the VDCNN model is the most accurate, but the GRU model displays more consistent results.

Installation
---
A working Python 3 installation is assumed. Install the required packages using:

```
pip install -r requirements.txt
```

Note that `requirements.txt` references the `tensorflow-gpu` package. It is recommended to use a GPU to train the models. If no GPU is used, install the `tensorflow` package instead.

Usage
---
Download the training data using:

```
./download.sh
```

Run the preprocessing steps using:

```
./process.sh
```

Now, you can train a model using:

```
./train.py -a vdcnn -d blogs pan13_tr_en
```

Use `train.py -h` for more information.
