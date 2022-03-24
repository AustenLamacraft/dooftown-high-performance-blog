---
title: "Machine Learning and Statistical Mechanics I"
subtitle: ""
summary: "TCM graduate lectures on ML"
authors: []
tags: []
categories: []
date: 2021-05-19T16:50:40+01:00
lastmod: 2021-05-19T16:50:40+01:00
featured: false
draft: false
projects: []
---


In these lectures we are going to explore some connections between machine learning (ML) and (classical) statistical mechanics (SM). To be precise, we are going to see how the appearance of *probabilistic models* with *large numbers of variables* in both fields means that certain theoretical concepts and tools can be applied in both. To get things going, let's see how this probabilistic viewpoint arises in the two settings.

$$
\DeclareMathOperator*{\E}{\mathbb{E}}
\newcommand{\cE}{\mathcal{E}}
$$

### Probability in Statistical Mechanics

The basic problem of SM is to describe the thermodynamic properties of  a macroscopic system *probabilistically* in terms its microscopic constituents. 


Note that we say *probabilistically*, not *statistically*. What's the difference? In probability we have a definite model for randomness in mind (the Boltzmann distribution, say), whereas in statistics we are interested in inferring these probabilities from observation. Statistics is thus [inverse probability](https://en.wikipedia.org/wiki/Inverse_probability), and statistical mechanics is really a misnomer.


The probabilistic model is normally the Boltzmann distribution 

$$
p(\mathbf{x})=\frac{\exp\left[-\beta \mathcal{E}(\mathbf{x})\right]}{Z},
$$

where $Z$ is a normalizing constant called the partition function, $\mathcal{E}(\mathbf{x})$ is the energy of the configuration $\mathbf{x}$, and $\beta=1/T$ is the inverse temperature (setting the Boltzmann constant $k_\text{B}=1$).

The *central problem* of statistical mechanics is computing ensemble averages of physical quantities, and the *principle difficulty* is the intractability of those averages for large systems. For example, if we are dealing with a classical gas, the configuration space point $\mathbf{x}$ corresponds to the positions of each of the gas molecules $\mathbf{x}=(\mathbf{x}_1,\ldots \mathbf{x}_N)$ and an average is a $3N$-dimensional integral. The only situation in which this integral is tractable is when the gas is noninteracting (ideal), in which case the energy function takes the form

$$
\mathcal{E}(\mathbf{x}) = \sum_{n=1}^N \mathcal{E}_1(\mathbf{x}_n)
$$

where $\mathcal{E}_1(\mathbf{x})$ is the single particle energy. In this case the integral factorizes. As soon as we introduce interactions between particles of the form

$$
\mathcal{E}(\mathbf{x}) = \sum_{n<m}^N \mathcal{E}_2(\mathbf{x}_n,\mathbf{x}_m)
$$

things get a lot harder. The same issue arises in models involving discrete random variables. The canonical example is the [Ising model](https://en.wikipedia.org/wiki/Ising_model), in which a configuration corresponds to fixing the values of $N$ "spins" $\sigma_n=\pm 1$ with an energy function of the form

$$
\mathcal{E}(\sigma)=\sum_n h_n\sigma_n + \sum_{n,m} J_{mn}\sigma_m\sigma_n.
$$

The two terms correspond to a (magnetic) field that acts on each spin and a coupling between spins. As in the gas, it's the latter that causes problems / interest. 

The Ising model comes in a great many flavours according to how the fields and couplings are chosen. They may reflect a lattice structure: $J_{mn}\neq 0$ for nearest neighbours, say, or longer range. They may be fixed or random, defining an ensemble of models. You've probably seen lots of examples already.

The most pessimistic assessment is that to calculate an average we are going to have sum over $2^N$ configurations. You probably know, however, that over the years physicists have developed lots of methods to solve the problem approximately, including mean field theory and Monte Carlo simulations. We'll return to this stuff later.

If you ever find yourself talking to a probabilist, you may find it helpful to know that these kind of models are called (undirected) [graphical models](https://en.wikipedia.org/wiki/Graphical_model), because their probability distribution is defined by a graph, called a [factor graph](https://en.wikipedia.org/wiki/Factor_graph). 

### Probability in Machine Learning 

What about ML? Let's take computer vision, one of the problems in which ML has made great progress in recent years. A (static) image is defined by a set of $(R,G,B)$ values at each pixel, each defined by eight bits i.e. an integer in $[0,255]$. The **basic hypothesis** on which probabilistic machine learning rests is that a dataset represents a set of independent and identically distributed (**iid**) samples of some random variables. In the case of an image, the random variables are the RGB values of all the pixels. The distribution of these variables has to be highly correlated and have a great deal of complex structure: rather than generating white noise for each sample we instead get (say) cats and dogs.

While this may seem like a funny way of thinking about a stack of photos it does conceptually have a lot in common with the way probability is often used in physics. After all, classical statistical mechanics is built on the notion that the motion of gas molecules is completely deterministic but incredibly complicated. While detailed knowledge of the dynamics is completely beyond our reach it is also irrelevant for the thermodynamic behaviour of interest: two boxes of gas behave in exactly the same way despite the underlying configurations of the molecules being completely different. Physics is used, however, to constrain our probability model. For example, collisions between molecules are elastic and momentum conserving.

The **difference** from the SM situation is that we don't know the probability distribution up front. The goal is to *infer* the distribution from data. Conceptually then, probabilistic ML is the same as [statistical inference](https://en.wikipedia.org/wiki/Statistical_inference). The different terms mostly reflect the differing background of practitioners: ML comes from computer science; statistical inference from mathematics. It all comes down to the tools you use: in recent years probabilistic ML has made great strides using models based on neural networks together with the associated training algorithms, which have allowed very rich probability distributions, describing datasets of images or audio signals, to be successfully modelled. 

<!-- Should note that the iid assumption is clearly wrong -->



<!-- The tone will be theoretical.

Applications of probablistic ML

- Sampling (**generative modelling**)
- Density estimation
- Compression

Physics uses. Sampling, better MC, etc, etc.


Mention possibility of *synthetic data*.

Planted ensembles

- [ ] Contrast Ising model and pics of faces
- [ ] Not going to discuss supervised learning, labels and all that
- [ ] Refer to Alemi here? -->


## Lecture 1: Fundamentals

### Some mathematical background

#### Probabilities: joint and conditional

Probabilities are real positive numbers $p(x)\geq 0$ satisfying 

$$
\begin{equation}
\sum_x p(x)=1
\end{equation}
$$

For continuous variables we have an integral of a probability density function

$$
\int p(x) dx=1,
$$

but for brevity we'll use the discrete notation throughout. 

**Joint probability** distributions of several variables are denoted $p(x_1,\ldots x_N)$. By summing over some subset of the variables, we arrive at a **marginal distribution** of the remaining variables:

$$
p(x)= \sum_{y} p(x,y).
$$

A related notion is the **conditional probability** $p(x|y)$: the probability distribution of $x$ given a fixed value of random variable $y$. The relation between joint and conditional probabilities is 

\begin{equation}
p(x,y)=p(x|y)p(y)
\label{eq:joint}
\end{equation}


We *should* write $p_X(x)$ for the distribution of random variable $X$ and $p_Y(y)$ for random variable $Y$. Instead, we just let the name of the argument tell us it's a different distribution. Everyone does this.


For a joint probability of many variables, we have


\begin{equation}
p(x_1,\ldots x_N)=p(x_1)p(x_2|x_1)p(x_3|x_2,x_1)\cdots p(x_N|x_1,\ldots x_{N-1}),
\label{eq:chain}
\end{equation}


which is sometimes called the [chain rule of probability](https://en.wikipedia.org/wiki/Chain_rule_(probability)). Although it's always possible *in principle* to express a joint probability like this, there's no guarantee it's easy to do or useful. One situation in which one may expect it to be a convenient description is when there is a natural order to the variables. For example, words or characters in text or any kind of time series. In this case the model may still be useful if the conditional probabilities involve only a fixed number $p$ of the preceding variables, even as $N\to\infty$. Such models are called [autoregressive](https://en.wikipedia.org/wiki/Autoregressive_model), although a physicist may be tempted to call them *causal*.

Sampling from a highly complex joint distribution $p(x_1,\ldots x_N)$ is generally difficult. One of the benefits of formulating a model as in \eqref{eq:chain} is that producing samples is much easier. First you sample $x_1$ using $p(\cdot)$, then sample $x_2$ using $p(\cdot|x_1)$, and so on. You never have to sample more than one variable at once!

#### Priors and posteriors

Another way to express the joint probability $\eqref{eq:joint}$ is

$$
p(x,y)=p(y|x)p(x)
$$

We deduce [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)

$$
p(y|x)=\frac{p(x|y)p(y)}{p(x)}
$$

Note that if we are dealing with continuous variables, any change in dimensions in going from a distribution over $x$ to a distribution over $y$ is handled by the ratio $p(y)/p(x)$.


Bayes' theorem is the workhorse of Bayesian statistics. The idea to to regard any parameters $z$ in your probability model as random variables taken from some initial distribution $p(z)$, called the **prior distribution** (or just the **prior**). 

> **Example**: if your model distribution is a Gaussian normal distribution with mean $\mu$ and variance $\sigma^2$, then your parameters are $z=(\mu,\sigma^2)$. For a prior you could choose a normal distribution for $\mu$ with its own mean $\mu_\mu$ and variance $\sigma^2_\mu$ (we write $\mu\sim \mathcal{N}(\mu_\mu,\sigma^2_\mu))$. For $\sigma^2$ you'd need a distribution of a positive quantity: the [inverse gamma distribution](https://en.wikipedia.org/wiki/Inverse-gamma_distribution) is a popular choice. 


Once these parameters are fixed, you have a model distribution for your data that can be thought of as the conditional distribution $p(x|z)$. What does an observation of $x$ tell me? Just use Bayes:

$$
p(z|x) = \frac{p(x|z)p(z)}{p(x)}.
$$

This is called the **posterior distribution** (or just **posterior**). Note that the denominator doesn't depend on $z$, it just provides a normalization. If you have lots of data points then

$$
p(z|x_1,\ldots x_N) \propto  p(x_1,\ldots x_N|z)p(z).
$$

Bayes' theorem lets us update our beliefs about parameters based on our initial beliefs and any evidence we receive. This process is called **inference**.

<!-- This feature of Bayesian statistics has led to a great many "thought leader" types describing themselves as Bayesians even though they wouldn't know their numerators from their denominators. -->


<!-- In the real world, it's not that easy. You don't actually get the data distribution $p(x)$, but rather samples from it. Also, your model may not be good enough for the task in hand. We'll talk about how we deal with these issues shortly. -->

#### Latent variables; generative models

Bayesian inference also underlies models involving *latent* (or hidden, or unobserved) variables. The idea here is that instead of the data telling us about the distribution of $z$s whose values may describe the entire dataset i.e. $p(x_1,\ldots x_N|z)$, we allow the $z$s to have different distributions for different data points $p(z_n|x_n)$. Equivalently, our model is defined by a joint distribution $p(x,z)$. 

The simplest example of a latent variable model is a [mixture model](https://en.wikipedia.org/wiki/Mixture_model), which describes a distribution of a variable $x$ as arising from $M$ different components, each with their own distribution $p(x|m)$ and occurring with probability $p(m)$, so that

$$
p(x) = \sum_m p(m)p(x|m).
$$

An observation $x$ will give me information about $p(m|x)$, telling which of the $M$ components that observation belongs to. This may bring insight, if the latent variables lend themselves to interpretation. Alternatively, it may simply give a more powerful model.

Latent variables are a route to perform **structure learning**: uncovering meaningful structures in the data. A lot of effort has gone into trying to build models that "discover" such structures automatically. For example, for a dataset of images of people walking we'd like to find latent variables parameterize a manifold of different poses.

Latent variable models are also the basis of **generative modelling**: sampling from a distribution $p(x)$ learnt from data. If the model has been formulated in terms of a prior $p(z)$ over latent variables and a generative model $p(x|z)$, sampling is straightforward in principle.

#### Entropy

In SM we're familiar with the entropy associated with a probability distribution. This quantity arrived in ML from information theory and is given the symbol $H$ (for [Hartley](https://en.wikipedia.org/wiki/Ralph_Hartley)?)

$$
H[p]=- \sum_x p(x)\log_2p(x).
$$

Taking the logarithm base 2 means we measure in bits (the natural logarithm that is normally used for the Gibbs entropy is measured in "nats"). In the following we'll normally drop the base.

There are lots of ways to think about the entropy so I'll just describe one that's quite useful in our setting. 
<!-- Suppose we have a $N$ samples from a uniform random distribution on $x$ (so strictly there will have to be a finite number of possible outcomes, but we can always take limits). What's the chance of observing the fractions $p(x)=N_x/N$? The chance of any set of values is just $1/|X|^N$, where $|X|$ is the number of possible outcomes, so to get the probability of the fractions $p(x)$ we multiply by the multinomial coefficient

$$
p(N_1,\ldots N_{|X|})= |X|^{-N}\frac{N!}{\prod_x N_x!}.
$$

Using Stirling's approximation $\log n! \sim n\log n -n$ gives, for large $N$

$$
p(N_1,\ldots N_{|X|})\sim|X|^{-N}2^{-N\sum_x p(x)\log_2 p(x)}=|X|^{-N}2^{NH[p]}.
$$


To see what $\sim$ means in this context note that when $p(x)=1/|X|$ (uniform distribution) the RHS is one, so this expression is missing a normalization factor (which you can get from a [better Stirling's approximation](https://en.wikipedia.org/wiki/Stirling%27s_approximation)), but that factor is not exponential in $N$, so we drop it.


 The entropy is therefore a useful measure of **exponentially unlikely events**. If we divide a container of $N$ ideal gas molecules up into $|X|$ identical regions $x=1\,\ldots |X|$, the probability of finding a fraction of gas $p(x)$ in each region is $\propto 2^{NH[p]}$.  -->
Suppose we have $N$ iid variables with distribution $p(x)$. The probability of observing a sequence $x_1,\ldots x_N$ is


\begin{equation}
p(x_1,\ldots x_N)=\prod_{n=1}^N p(x_n).
\label{eq:seq}
\end{equation}


This probability is obviously exponentially small as $N\to\infty$, but how small? The answer is 

$$
\lim_{N\to\infty} \frac{1}{N}\log p(x_1,\ldots x_N) = -H[p].
$$

This is called the [asymptotic partition property](https://en.wikipedia.org/wiki/Asymptotic_equipartition_property). It probably looks a bit strange. Shouldn't the probability depend on what you actually get? After all, some outcomes *are* more likely than others. Suppose you have a biased coin that gives heads with probability $p_H>0.5$ and tails with probability $p_T=1-p_H$. In a very long sequence of tosses the chance of getting half heads and half tails becomes exponentially small. What you're going to get instead is

$$
\frac{N_H}{N}\to p_H\qquad \frac{N_T}{N}\to p_T\qquad .
$$

What is the probability of such a sequence? From $\eqref{eq:seq}$ it is $p_H^{N_H}p_T^{N_T}$, but this can be rewritten

$$
\log_2\left(p_H^{N_H}p_T^{N_T}\right)= N_H\log_2 p_H + N_T\log_2 p_T = -N H[p_H, p_T]. 
$$

#### Entropy and information 

This property of entropy provides a way to quantify the amount of information in a signal. If the coin is *really* biased, returning a head almost every time, you won't be surprised when you get heads, but will be surprised when you get tails. Note that the entropy of such a sequence is lower than for a fair coin, which has the maximum entropy $H=1$ . If you wanted to describe such sequence, like

> HHHHHHHHHHHHHHHHHHHHHTHHHHHHHHHHHHHTHHHHT


you might find yourself saying something like "21 H, 13 H, 4 H", with the understanding that between each string of heads there's one tail. Such a description is shorter than the original sequence, which is possible because of the high degree of predictability. This isn't a like for like comparison, however, because we've introduced extra symbols including the digits 0-9 and some delimiter like the comma. We should instead compare with a binary code of only two symbols. How can we exploit the lower entropy of the sequence to come up with an encoding that is better than the "literal" one? One (not very practical) way is to use the fact that we expect $N_H=Np_H$ heads and $N_T=N p_T$ tails, so we can just give the *ordering* of these heads and tails, which is one of

$$
\frac{N!}{N_H! N_T!}
$$

possibilities. If we label each of these with a binary number, we end up with a description of length

$$
\log_2\left(\frac{N!}{N_H! N_T!}\right)\sim N H[p]\leq N
$$

(where we used Stirling's approximation $\log n! \sim n\log n -n$). Now of course, we are unlikely to get exactly this number of heads and tails, but correcting for this requires a number of bits that can be neglected in the large $N$ limit (i.e. it is $o(N)$).

This example is the simplest illustration of [Shannon's source coding theorem](https://en.wikipedia.org/wiki/Shannon%27s_source_coding_theorem):

> N i.i.d. random variables each with entropy H(X) can be compressed into more than N H(X) bits with negligible risk of information loss, as N → ∞; but conversely, if they are compressed into fewer than N H(X) bits it is virtually certain that information will be lost.

Shannon's theorem is the core idea that underlies (lossless) data compression: the more predictable a signal (i.e. the lower the entropy) the more it can be compressed, with the entropy setting a fundamental limit on the number of bits required. 

How is this idea applied in the real world? It's probably clear that real binary data doesn't have a preponderance of 1s or 0s, that would obviously be inefficient. It all hinges on what you regard as your iid random variables. For example, text consists of strings of characters, of which there are 143,859 in the [Unicode standard](https://home.unicode.org/), including scripts from different languages and Screaming Cat 🙀. These obviously don't occur with the same frequency, so the entropy is going to be much less than $\log_2(143859)\approx 17.1$ bits per character. Very roughly, a compression scheme involves choosing short codewords for common characters and long codewords for rare characters. Immediately you'll notice there's an issue in deciding where one character ends and the next begins. If you're interested in how this works in practice, see [Huffman coding](https://en.wikipedia.org/wiki/Huffman_coding), [arithmetic coding](https://en.wikipedia.org/wiki/Arithmetic_coding), and [asymmetric numeral systems](https://en.wikipedia.org/wiki/Asymmetric_numeral_systems), or the book  [Understanding Compression](https://www.oreilly.com/library/view/understanding-compression/9781491961520/).

The bigger issue, however, is that text *doesn't* consists of iid characters, even if drawn with the right frequencies. A [Markov model](https://en.wikipedia.org/wiki/Markov_model) would be the natural next step, in which the probability of each character is conditional on the preceding character: $p(x_{n+1}|x_{n})$. You're likely (in English) to encounter a `u` after a `q`, for instance. Next you can go to a "second order" Markov model, with $p(x_{n+1}|x_{n}, x_{n-1})$, and so on. [Shannon's original paper](https://ieeexplore.ieee.org/document/6773067) is wonderfully clear and provides examples of experiments on these models. He calls the character frequency model "first order" and the Markov model "second order".

What really matters, then, is *how good your model is*. Suppose you want to compress data that consists of (unrelated) images. Each image is described by the RGB values of all the pixels: we'll denote these values collectively by $\mathbf{x}$. If you choose your encoding according to some model probabilities $p_\text{M}(\mathbf{x})$, the encoding of image $\mathbf{x}$ will have length $-\log_2\left[p_\text{M}(\mathbf{x})\right]$ bits. When you encode your data you get on average

$$
-\frac{1}{N}\sum_{\mathbf{x}} \log_2\left[p_\text{M}(\mathbf{x})\right] = -\sum_{\mathbf{x}} p_\text{D}(\mathbf{x})\log_2\left[p_\text{M}(\mathbf{x})\right]
$$

bits per image, where $p\_\text{D}(\mathbf{x})$ is the **empirical distribution** of the data. The quantity on the RHS is called the [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) (or relative entropy) $H(p_\text{D}, p_\text{M})$. As we'll see in the next section, it has the key property that is it bounded from below by the entropy $H(p_\text{D})$

$$
H(p_\text{D}, p_\text{M})\geq H(p_\text{D}).
$$

The trade-off, then, is 

- By considering bigger chunks of data you approach closer to the iid situation to which Shannon's theorem applies, **but**
- These big chunks (images in our example) will have correspondingly more complicated distributions $p_\text{D}$, which your model $p_\text{M}$ will have to match if you want to approach optimal encoding.


#### Divergences

The above discussion should make it clear that we need some way of talking about the degree to which two distributions differ. The most common measure in use in ML is the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) (KL)

$$
D_\text{KL}(p||q)=\sum_x p(x)\log\left(\frac{p(x)}{q(x)}\right)=\E_{x\sim p}\log\left(\frac{p(x)}{q(x)}\right).
$$


More notation. $\E$ denotes the expectation and $x\sim p$ means that $x$ follows the distribution $p$. Thus
$$
\E_{x\sim p}\left[f(x)\right]=\sum_x p(x)f(x)
$$


It's not hard to show that the KL is related to the cross entropy we just introduced by

$$
H(p,q)= D_\text{KL}(p||q)+ H(p).
$$

Thus the statement that the cross entropy $H(p,q)$ is bounded from below by the entropy $H(p)$ is equivalent to the KL being non-negative

$$
D_\text{KL}(p||q)\geq 0
$$

This is simple consequence of [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality), which is the statement that for a convex function $\varphi(x)$

$$
\E\left[\varphi(x))\right]\geq \varphi\left(\E\left[x\right]\right)
$$

If we apply this to the KL then, using the convexity of $\varphi(x)=-\log(x)$

$$
D_\text{KL}(p||q)=-\E_{x\sim p}\log\left(\frac{q(x)}{p(x)}\right)\geq -\log\left(\E_{x\sim p}\left[\frac{q(x)}{p(x)}\right]\right)=-\log(1)=0,
$$

with equality if and only if $p=q$. 

<!-- By the way, the term **divergence** is used to distinguish from a **distance**, which is symmetric and satisfies the triangle inequality.



KL page discusses relation to coding

[Sanov theorem](https://en.wikipedia.org/wiki/Sanov%27s_theorem) but Wikipedia page isn't great.

Divergences, KL divergences. Properties. 

Gibbs inequality. ELBO

How compressed is your signal in practice? KL



Minimal description length

- [ ] Forward and reverse KL and the meaning -->

### Variational inference (VI)

After introducing Bayes' theorem we discussed how you might go about fitting a model to data. It's not as easy we made it sound. Recall that Bayes' says the posterior distribution is

$$
p(z|x) = \frac{p(x|z)p(z)}{p(x)}=\frac{p(x,z)}{p(x)}.
$$

The denominator $p(x)=\sum_z p(x,z)$ normalizes the distribution $p(z|x)$, just like the partition function of a SM model. If we are dealing with a complicated latent variable model where $z$ and $x$ are both high dimensional, and $p(x,z)$ has a complex structure, this is intractable.

<!-- The problems are

1. We don't actually have $p(x)$, just a dataset that we interpret as providing samples from $p(x)$.

2. If the data is highly structured and complex (keep our images example in mind) then $p(x|z)$ will have to be a similarly complicated model if it's going to have a chance of success. Think of the Ising model with all sorts of nasty couplings and fields. To evaluate $p(x|z)$ we are going to have to calculate the normalization factor aka the partition function $Z$, and that's going to be hard for a big model. -->

In this section we'll see that it's possible to develop a variational formulation of the problem that returns the "best" posterior given a family of models. It's basically a straight copy of physicists' mean field theory, so we'll review that first using the language of probability.

#### Mean field theory

For an SM model like the Ising model the probability has the form


\begin{equation}
p(\sigma) = \frac{\exp\left[-\beta\cE(\sigma)\right]}{Z}.
\label{eq:boltzmann}
\end{equation}


The goal is to find expectations, for example the average energy $\E\_{\sigma\sim p}\left[\cE(\sigma)\right]$. Since this is difficult for the $p(\sigma)$ nature gives us we are going to try and approximate $p(\sigma)$ by a *simpler* class of distributions $q\_\phi(\sigma)$, where $\phi$ denote the parameters that define the family, and find the *best* approximation.

What does *simpler* mean? It means one where we can actually calculate expectations (with the resources we have available). Probably the most drastic simplification we can take is to suppose that the variables are independent, so that the probability distribution factorizes

\begin{equation}
q_\phi(\sigma)=\prod_n q_{\phi_n}(\sigma_n).
\label{eq:factor}
\end{equation}

We are allowing for the single spin distributions to be different, which will be appropriate for an inhomogeneous model, the kind of thing you would use to describe a disordered spin system.

What does *best* mean? We've seen that the KL quantifies the difference between distributions, so it's natural to try to minimize 

$$
D_\text{KL}(q||p)=\E_{\sigma\sim q_\phi}\left[\log\left(\frac{q_\phi(\sigma)}{p(\sigma)}\right)\right].
$$

Why do we minimize $D(q||p)$ and not $D(p||q)$? The pragmatic answer is: *$D(q||p)$ is the one we can calculate*, as it involves an expectation with respect to the tractable trial distribution. Substituting in the Boltzmann distribution $\eqref{eq:boltzmann}$ we find

$$
D_\text{KL}(q||p)= \log Z - H[q_\phi] + \beta \E_{\sigma\sim q_\phi}\left[\cE(\sigma)\right]\geq 0,
$$

or in usual SM language

\begin{equation}
\E_{\sigma\sim q_\phi}\left[\cE(\sigma)\right]-TH[q_\phi] \geq F,
\label{eq:mft}
\end{equation}

where $F=-T\log Z$ is the Helmholtz free energy. This is known as variously as the [Bogoliubov](https://en.wikipedia.org/wiki/Helmholtz_free_energy#Bogoliubov_inequality) or [Gibbs](https://en.wikipedia.org/wiki/Gibbs%27_inequality) inequality. By optimizing the left hand side over $\phi$ we can find the best approximation within our family, and it will achieve a free energy closest to the true value.

For Ising spins our factorized distributions $\eqref{eq:factor}$ are defined by fields on each site

$$
q_{\phi_n}(\sigma_n) = \frac{\exp\left[-\beta\phi_n\sigma_n\right]}{2\cosh (\beta\phi_n)}, 
$$

with average spin

$$
\E_{\sigma_n\sim q_n}\left[\sigma_n\right] = -\tanh\left(\beta\phi_n\right).
$$

Optimizing $\eqref{eq:mft}$ with respect to $\phi_n$ reproduces the equations of [mean field theory](https://en.wikipedia.org/wiki/Mean-field_theory#Ising_model). The optimal values of $\phi_n$ are interpreted as the "mean fields" due to the applied field and the other spins coupled to $\sigma_n$.

#### VI in latent variable models

The only thing we need to do to apply the same idea to latent variable models is to replace $\eqref{eq:boltzmann}$ with 

$$
p(z|x) =\frac{p(x,z)}{p_\text{M}(x)}.
$$

(we add the subscript "M" for model) The role of the spins $\sigma$ is now played by the latent variables. Following exactly the same steps leads us to 

\begin{equation}
\log p_\text{M}(x) \geq \E_{z\sim q_\phi(\cdot|x)}\left[\log p(x,z)\right]+ H[q_\phi(\cdot|z)].
\label{eq:elbo}
\end{equation}

The right hand side is called the [Evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) or **ELBO** (because the marginalized probability $p(x)$ on the left is sometimes called the **model evidence**). 

It's possible to re-write $\eqref{eq:elbo}$ as

$$
\log p_\text{M}(x) \geq \log p_\text{M}(x) - D_\text{KL}(q_\phi(\cdot|x)||p(\cdot|x)),
$$

so the bound is saturated when the variational posterior for the latent variables coincides with the true posterior $p(z|x)=p(x,z)/p_\text{M}(x)$. 

For complicated models it isn't practical to optimize the ELBO for each data point $x$ to obtain the best $\phi(x)$. Instead, we average over the entire data set (this is called *amortization*, a pretty confusing term). I like to think of this as follows. We have two representations of the same joint distribution. One ("forward") in terms of the generative model and the prior

$$
p_\text{F}(x,z)= p(x|z)p(z)
$$

and the other ("backward") in terms of the data distribution $p_\text{D}(x)$ and the posterior

$$
p_\text{B}(x,z)= q_\phi(z|x)p_\text{D}(x).
$$

To make the two equal we should minimize the KL over joint distributions

$$
D_\text{KL}(q||p)(p_\text{B}||p_\text{F})=\E_{x\sim \text{Data}}\left[\E_{z\sim q_\phi(\cdot|x)}\left[\log\left(\frac{q_\phi(z|x)p_\text{D}(x)}{p(x|z)p(z)}\right)\right]\right]\geq 0,
$$

or

$$
H[p_\text{D}]\leq H(p_\text{D}, p_\text{M}) \leq \E_{x\sim \text{Data}}\left[\E_{z\sim q_\phi(\cdot|x)}\left[\log\left(\frac{q_\phi(z|x)}{p(x|z)p(z)}\right)\right]\right].
$$

By improving our posterior we can saturate the second equality. By improving our generative model $p(x|z)$ we can saturate the first. 

A popular modern approach is to introduce models $q_\phi(x|z)$ and $p_\theta(z|x)$, often parameterized in terms of neural networks, and optimize both sets of parameters $\theta$ and $\phi$ simultaneously. This is the basis of the [Variational Autoencoder](https://en.wikipedia.org/wiki/Autoencoder#Variational_autoencoder_(VAE)), which we'll meet in the next lecture.

<!-- #### A toy model


Community detection
Other stuff. Message passing

Example of planted models

### A zoo of models

Graphical models, Markov models, autoregressive models

Hidden Markov models

Examples from applications

### Formalising the problem

See Alemi paper "the world we want" 



## Lecture 2: Models

### Variational autoencoder

Continuous *vs.* discrete variables.

Bits back encoding

### Normalizing flows

Relation to VAEs

### Diffusion models

### Optimal Importance Sampling

Jarzynski inequality. Annealed importance sampling

### Schrodinger bridge

Turning a model into an autoregressive model -->