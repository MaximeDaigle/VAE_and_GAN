# VAE and GAN

Content
1. Implementation of a **VAE**
2. Comparison of a GAN trained with Squared Hellinger distance vs Wasserstein distance 
3. Implementation of a **WGAN**
4. Training of the WGAN on Street View House Numbers

#### [Bonus theoretical content](https://github.com/MaximeDaigle/VAE_and_GAN/blob/main/theoretical_generative.pdf) touching:
* Autoregressive models 
* Reparameterization trick 
* Variational autoencoders (VAE)
* Normalizing flows 
* Generative adversarial networks (GANs)



## VAE

Variational Autoencoders (VAEs) are probabilistic generative models to model data distribution p(x). In this section, a VAE is trained on the Binarised MNIST dataset, using the negative ELBO loss. Note that each pixel in this image dataset is binary: The pixel is either black or white, which means each datapoint (image) is a collection of binary values. The likelihood p<sub>&theta;</sub>(x|z), i.e. the decoder, is modelized as a product of bernoulli distributions.


## GAN's Comparison
Generative Adversarial Network (GAN) enables the estimation of distributional measure between arbitrary empirical distributions. This Section implements a function to estimate the Squared Hellinger as well as one to estimate the Earth mover distance. This allows to look at and contrast some properties of the [f-divergence](https://arxiv.org/abs/1606.00709) and the Earth-Mover distance ([Wasserstein GAN](https://arxiv.org/abs/1701.07875)).

#### squared hellinger

<img src="https://github.com/MaximeDaigle/VAE_and_GAN/blob/main/images/squared_hellinger.png" alt="squared hellinger" width="650"/>

#### Wasserstein

<img src="https://github.com/MaximeDaigle/VAE_and_GAN/blob/main/images/Wasserstein.png" alt="wasserstein" width="550"/>

<img src="https://github.com/MaximeDaigle/VAE_and_GAN/blob/main/images/Lipschitz.png" alt="lipschitz" width="550"/>

#### Comparison

![comparison](https://github.com/MaximeDaigle/VAE_and_GAN/blob/main/images/comparison.png)


## Training

The sequential language models are trained on the
Penn Treebank dataset. Language models learn to assign a likelihood to
sequences of text. The elements of the sequence (typically words or
individual characters) are called tokens, and can be represented as
one-hot vectors with length equal to the vocabulary size, e.g. 26 for a
vocabulary of English letters with no punctuation or spaces, in the case
of characters, or as indices in the vocabulary for words. In this
representation an entire dataset (or a mini-batch of examples) can be
represented by a 3-dimensional tensor, with axes corresponding to: (1)
the example within the dataset/mini-batch, (2) the time-step within the
sequence, and (3) the index of the token in the vocabulary. Sequential
language models do next-step prediction, in other words, they
predict tokens in a sequence one at a time, with each prediction based
on all the previous elements of the sequence. A trained sequential
language model can also be used to generate new sequences of text, by
making each prediction conditioned on the past *predictions* (instead of
the ground-truth input sequence).

#### The Penn Treebank Dataset

This is a dataset of about 1 million words from about 2,500 stories from
the Wall Street Journal. It has Part-of-Speech annotations and is
sometimes used for training parsers, but it's also a very common
benchmark dataset for training RNNs and other sequence models to do
next-step prediction.

#### Preprocessing

The version of the dataset you will work with has been preprocessed:
lower-cased, stripped of non-alphabetic characters, tokenized (broken up
into words, with sentences separated by the `<eos>` (end of sequence)
token), and cut down to a vocabulary of 10,000 words; any word not in
this vocabulary is replaced by `<unk>`. For the transformer network,
positional information (an embedding of the position in the source
sequence) for each token is also included in the input sequence.

#### Loss

Unlike in classification problems, where the performance metric is
typically accuracy, in language modelling, the performance metric is
typically based directly on the cross-entropy loss, i.e. the negative
log-likelihood ($NLL$) the model assigns to the tokens. For word-level
language modelling it is standard to report **perplexity (PPL)**, which
is the exponentiated average per-token NLL (over all tokens):

<img src="https://github.com/MaximeDaigle/transformer-scratch/blob/main/images/ppl_eq.png" alt="ppl eq" width="400"/>

where t is the index with the sequence, and n indexes different
sequences. For Penn Treebank in particular, the test set is treated as a
single sequence (i.e. N=1). The purpose of this part is to perform
model exploration.

#### Results

The three architectures are trained using either stochastic gradient
descent or the ADAM optimizer. The training loop is provided in
*run\_exp.py*. For each experiment (3.1, 3.2, 3.3, 3.4), the learning
curves (train and validation) of PPL over both epochs and
wall-clock-time are in the folder images.

![Best Validation PPL for each experiment](images/table_result.png)
Best Validation PPL for each experiment


## Comparison of generated samples 

![generation_eq1](https://github.com/MaximeDaigle/transformer-scratch/blob/main/images/generation_eq.png)
