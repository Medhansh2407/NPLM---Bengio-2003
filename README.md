# NPLM — Bengio et al. (2003) | From-Scratch PyTorch Implementation

A from-scratch PyTorch implementation of the **Neural Probabilistic Language Model** (Bengio et al., 2003) trained on the Tiny Shakespeare dataset.

> Full paper walkthrough and implementation blog post: https://medhanshnarang.vercel.app/papers/a-neural-probabilistic-language-model-bengio-et-al-2003

---

## Why this paper?

Bengio et al. (2003) was a landmark. At a time when language models were n-gram based, this paper introduced **dense word embeddings** — representing words as continuous vectors in a shared feature space. That idea quietly became the foundation of modern NLP (Word2Vec, GloVe, transformers).

It solved three core problems with n-gram models:

- **Curse of dimensionality** — n-gram models explode in size as context grows. Trigrams over a 27-character vocab give 729 combinations; scale to words and it's completely intractable.
- **No notion of word similarity** — n-gram models treat every word as an independent symbol with no relationship to any other.
- **Zero probability for unseen sequences** — if a sequence wasn't in training data, probability is 0. Smoothing was a patch, not a fix.

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
