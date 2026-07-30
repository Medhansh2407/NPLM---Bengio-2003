# NPLM — Bengio et al. (2003) | From-Scratch PyTorch Implementation


This is a grounds up implementation of the ***Neural Probabilistic Language Model*** introduced 
in the paper(Bengio et al. , 2003) . The model is trained on the Tiny Shakespeare dataset.


> Full paper walkthrough and implementation blog post: https://medhanshnarang.vercel.app/papers/a-neural-probabilistic-language-model-bengio-et-al-2003

---

## Why this paper?

So this paper was an innovative  breakthrough of its time , it changed the the trajectory of 
language models from n-gram models to *dense word embeddings* , which represents words as 
continuous feature vectors and so this idea became the foundation of the later papers like Word2Vec , GloVe and transformers.


So the three core n-gram problems which this model solved were:

- **Curse of dimensionality** — n-gram models explode exponentially in size as the context grows , for example a bigram model which has the vocabulary of 27 characters of which the 26 characters are the English alphabets and one character is the special ending token . Now if we
we use a bigram model to perform this job , we would have one of the 27 characters as a context
character and would have to predict one of the 27 characters , which would mean dealing with
27C1 * 27C1 characters = 729 characters
now scale this up to trigrams and you would have to select one combination of 2 characters from 
729 possible combinations and then would have to select one of the 27 characters as the next character , well this was curse of dimensionality for you.
- **No notion of word similarity** —n-gram models only evaluate on the order of words meanwhile 
there is way more information in word sequence than just the word order , and so the NPLM was successful in working through those information by representing words through some n dimensional feature vector.
- **Zero probability for unseen sequences** — In the n gram models , an unseen sequence was 
assigned a 0 probability and so a fix for this was a mathematical manipulation and not the model 
actually learning through the patters and so smoothing was a patch not a fix to this problem.

The NPLM addresses all three through a learned embedding space and a neural network probability model.

---

## Architecture

```
Input: context window of previous words
  → Embedding lookup C: each word → 100-dim vector
  → Concatenate → hidden layer (tanh, 200 neurons)
  → Output: softmax over vocab
```

| Hyperparameter | Value |
|----------------|-------|
| Embedding dim  | 100   |
| Context window | 10    |
| Hidden neurons | 200   |
| Vocab size     | ~25k  |

---

## Setup

Download [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) and save as `input.txt` in the same folder as the notebook.

**Dependencies:** `torch`, `matplotlib`

Run all cells top to bottom.


---

## Paper

> Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003).  
> *A Neural Probabilistic Language Model.* JMLR, 3, 1137–1155.  
> https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
