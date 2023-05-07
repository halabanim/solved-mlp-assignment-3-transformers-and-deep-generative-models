Download Link: https://assignmentchef.com/product/solved-mlp-assignment-3-transformers-and-deep-generative-models
<br>
This assignment contains two parts. First, you will take a closer look at the Transformer architecture. They have been already introduced in Lecture 9 and <a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html">Tutorial</a> <a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html">6</a><a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html">,</a> but here, we will discuss more about the theoretical and practical aspects of the architecture.

The second part, which is the main part of the assignment, will be about Deep

Generative Models. Modelling distributions in high dimensional spaces is difficult. Simple distributions such as multivariate Gaussians or mixture models are not powerful enough to model complicated high-dimensional distributions. The question is: How can we design complicated distributions over high-dimensional data, such as images or audio? In concise notation, how can we model a distribution <em>p</em>(<em>x</em>) = <em>p</em>(<em>x</em><sub>1</sub><em>,x</em><sub>2</sub><em>,…,x<sub>M</sub></em>), where <em>M </em>is the number of dimensions of the input data <em>x</em>? The solution: Deep Generative Models.

Deep generative models come in many flavors, but all share a common goal: to model the probability distribution of the data. Examples of well-known generative models are Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs). In this assignment, we will focus on VAEs [Kingma and Welling, 2014]. The assignment guides you through the theory of a VAE with questions along the way, and finally you will implement a VAE yourself in PyTorch as part of this assignment. Note that, although this assignment does contain some explanation on the model, we do not aim to give a complete introduction. The best source for understanding the models are the lectures, these papers [Goodfellow et al., 2014, Kingma and Welling, 2014, Rezende and Mohamed, 2015], and the hundreds of blog-posts that have been written on them ever since.

Throughout this assignment, you will see a new type of boxes between questions, namely Food for thought boxes. Those contain questions that are helpful for understanding the material, but are not essential and not required to submit in the report (no points are assigned to those question). Still, try to think of a solution for those boxes to gain a deeper understanding of the models.

This assignment contains 50 points: 10 on Transformers, and 40 on VAEs. Only the VAE part contains an implementation task at the end.

Note: for this assignment you are <u>not </u>allowed to use the torch.distributions package. You are, however, allowed to use standard, stochastic PyTorch functions like torch.randn and torch.multinomial, and all other PyTorch functionalities (especially from torch.nn). Moreover, try to stay as close as your can to the template files provided as part of the assignment.

<h1>1           Attention and Transformers</h1>

In this part, we will discuss theoretical questions with respect to the Transformer architecture. Make sure to check the lecture on Transformers. It is recommended to also check the UvA Deep Learning Tutorial 6 on <a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html">Transformers and Multi-Head Attention </a>before continuing with this part.

<h2>1.1           Attention for Sequence-to-Sequence models</h2>

Conventional sequence-to-sequence models for neural machine translation have a difficulty to handle long-term dependencies of words in sentences. In such models, the neural network is compressing all the necessary information of a source sentence into a fixedlength vector. Attention, as introduced by Bahdanau et al. [2015], emerged as a potential solution to this problem. Intuitively, it allows the model to focus on relevant parts of the input, while decoding the output, instead of compressing everything into one fixed-length context vector. The attention mechanism is shown in Figure 1. The complete model comprises an encoder, a decoder, and an attention layer.

Unlike conventional sequence-to-sequence, here the conditional probability of generating the next word in the sequence is conditioned on a distinct context vector <em>c<sub>i </sub></em>for each target word <em>y<sub>i</sub></em>. In particular, the conditional probability of the decoder is defined as:

<em>p</em>(<em>y<sub>i</sub></em>|<em>y</em><sub>1</sub><em>,…,y<sub>i</sub></em><sub>−1</sub><em>,x</em><sub>1</sub><em>,…,x<sub>T</sub></em>) = <em>p</em>(<em>y<sub>i</sub></em>|<em>y<sub>i</sub></em><sub>−1</sub><em>,s<sub>i</sub></em>)<em>,                                                   </em>(1)

where <em>s<sub>i </sub></em>is the RNN decoder hidden state for time i computed as <em>s<sub>i </sub></em>= <em>f</em>(<em>s<sub>i</sub></em><sub>−1</sub><em>,y<sub>i</sub></em><sub>−1</sub><em>,c<sub>i</sub></em>).

The context vector <em>c<sub>i </sub></em>depends on a sequence of annotations (<em>h</em><sub>1</sub><em>,</em>··· <em>,h<sub>T</sub></em><em><sub>x</sub></em>) to which an encoder maps the input sentence; it is computed as a weighted sum of these annotations

<em>h<sub>i</sub></em>:

<em>T<sub>x</sub></em>

<table width="515">

 <tbody>

  <tr>

   <td width="498"><em>c</em><em>i </em>= X<em>α</em><em>ij</em><em>h</em><em>j</em><em>.</em><em>j</em>=1The weight <em>α<sub>ij </sub></em>of each annotation <em>h<sub>j </sub></em>is computed by</td>

   <td width="17">(2)</td>

  </tr>

 </tbody>

</table>

<em>,                                                             </em>(3)

where <em>e<sub>ij </sub></em>= <em>a</em>(<em>s<sub>i</sub></em><sub>−1</sub><em>,h<sub>j</sub></em>) is an alignment model which scores how well the inputs around position <em>j </em>and the output at position <em>i </em>match. The score is based on the RNN hidden state <em>s<sub>i</sub></em><sub>−1 </sub>(just before emitting <em>y<sub>i</sub></em>, Eq. (1)) and the <em>j</em>-th annotation <em>h<sub>j </sub></em>of the input sentence. The alignment model <em>a </em>is parametrized as a feedforward neural network which is jointly trained with all the other components of the proposed system. Use the Figure 1 to understand the introduced notations and the derivation of the equations above.

<h2>1.2           Transformer</h2>

Transformer is the first encoder-decoder model based solely on (self-)attention mechanisms, without using recurrence and convolutions. The key concepts for understanding Transformers are: queries, keys and values, scaled dot-product attention, multi-head and self-attention.

Queries, Keys, Values The Transformer paper redefined the attention mechanism by providing a generic definition based on queries, keys, values. The encoded representation of the input is viewed as a set of key-value pairs, of input sequence length. The previously generated output in the decoder is denoted as a query.

Figure 1. The graphical illustration of the attention mechanism proposed in [Bahdanau et al., 2015]

Scaled dot-product attention            In [Vaswani et al., 2017], the authors propose <em>scaled dot-product attention</em>. The input consists of queries and keys of dimension <em>d<sub>k</sub></em>, and values of dimension <em>d<sub>v</sub></em>. The attention is computed as follows:

Attention(<em>Q,K,V </em>) = softmax(                                             (4)

From this equation, it can be noticed that the dot product of the query with all keys represents a matching score between the two representations. This score is scaled by, because dot products can grow large in magnitude due to long sequence inputs. After applying a softmax we obtain the weights of the values.

Multi-head attention Instead of performing a single attention function, it is beneficial to linearly project the Q, K and V, h times with different, learned linear projections to <em>d<sub>k</sub></em>, <em>d<sub>k </sub></em>and <em>d<sub>v </sub></em>dimensions, respectively. The outputs are concatenated and once again projected by an output layer:

MultiHead(<em>Q,K,V </em>) = Concat(head<sub>1</sub><em>,…,</em>head<sub>h</sub>)<em>W<sup>O</sup>,</em>

where head<sub>i </sub>= Attention(<em>.</em>

Self-attention       Self-attention (also called intra-attention) is a variant of the attention mechanism relating different positions of a single sequence in order to compute a revised representation of the sequence.

<h3>Question 1.1</h3>

Consider the encoder-decoder attention introduced in Bahdanau et al. [2015] and the self-attention used in encoder and decoder of the Transformers architecture [Vaswani et al., 2017]. Compare the two mechanisms by denoting what the queries, keys and values represent. (Words limit: 100)

<h3>Question 1.2</h3>

Discuss the challenge of long input sequence lengths for the Transformer model by:

<ul>

 <li>Explaining what is the underlying cause of this challenge.</li>

 <li>Describing a way on how to overcome this challenge. (Words limit: 150)</li>

</ul>

<h2>1.3           Transformer-based Models</h2>

Transformers became the de-facto standard not only for NLP tasks but also for vision and multimodal tasks. The family of Transformer-based models grows every day, thus it is important to understand the fundamentals of different models and which tasks they can solve.

<h3>Question 1.3</h3>

Describe how would you solve the task of <em>Spam classification </em>by using Transformers, if no meta-data is available (only the email text itself). Specifically:

<ul>

 <li>Explain how the input and output representations should look like.</li>

 <li>Design the training and inference stage.</li>

</ul>

(Words limit: 200)

<h1>2           Variational Auto Encoders</h1>

VAEs leverage the flexibility of neural networks (NN) to learn and specify a latent variable model. We will first briefly discuss Latent Variable Models and then dive into VAEs. Mathematically, they are connected to a distribution <em>p</em>(<em>x</em>) over <em>x </em>in the following way: <em>p</em>(<em>x</em>) = <sup>R </sup><em>p</em>(<em>x</em>|<em>z</em>)<em>p</em>(<em>z</em>)<em>d</em><em>z</em>. This integral is typically too expensive to evaluate. However, in this assignment, you will learn a solution via VAEs.

<h2>2.1           Latent Variable Models</h2>

<table width="517">

 <tbody>

  <tr>

   <td width="401">                                                    <em>n                              </em><em>D</em><em>x<sub>n </sub></em>∼ <em>p<sub>X</sub></em>(<em>f<sub>θ</sub></em>(<em>z</em><em><sub>n</sub></em>))                                               (6)where <em>f<sub>θ </sub></em>is some function – parameterized by <em>θ </em>– that maps <em>z</em><em><sub>n</sub></em></td>

   <td width="115">Figure 2. Graphical model of VAE. <em>N </em>denotes the dataset size.</td>

  </tr>

 </tbody>

</table>

A latent variable model is a statistical model that contains both observed and unobserved (i.e. latent) variables. Assume a dataset , where <em>x</em><em><sub>n </sub></em>∈ {0<em>,</em>1<em>,…,k </em>− 1}<em><sup>M</sup></em>. For example, <em>x</em><em><sub>n</sub></em>

could be the pixel values for an image, in which each pixel can take values 0 through <em>k </em>− 1 (for example, <em>k </em>= 256). A simple latent variable model for this data is shown in Figure 2, which we can also summarize with the following generative story:

<em>z </em>∼ N(0<em>,</em><em>I </em>)                                                    (5)

to the parameters of a distribution over <em>x</em><em><sub>n</sub></em>. For example, if <em>p<sub>X </sub></em>would be a Gaussian distribution we will use  for a mean and covariance matrix, or if <em>p<sub>X </sub></em>is a product of Bernoulli distributions, we have <em>f<sub>θ </sub></em>: R<em><sup>D </sup></em>→ [0<em>,</em>1]<em><sup>M</sup></em>. Here, <em>D </em>denotes the dimensionality of the latent space. Likewise, if pixels can take on k discrete values, <em>p<sub>X </sub></em>could be a product of Categorical distributions, so that <em>f<sub>θ </sub></em>: R<em><sup>D </sup></em>→ (<em>p</em><sub>1</sub><em>,…,p<sub>k</sub></em>)<em><sup>M</sup></em>. Where <em>p</em><sub>1</sub><em>,…,p<sub>k </sub></em>are event probabilities of the pixel belonging to value <em>k</em>, where <em>p<sub>i </sub></em>≥ 0 and. Note that our dataset D does not contain <em>z</em><em><sub>n</sub></em>, hence <em>z</em><em><sub>n </sub></em>is a latent (or unobserved) variable in our statistical model. In the case of a VAE, a (deep) NN is used for <em>f<sub>θ</sub></em>(·).

<h3>Food for thought</h3>

How does the VAE relate to a standard autoencoder (see e.g. <a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html">Tutorial</a> <a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html">9</a><a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html">)</a>?

<ol>

 <li>Are they different in terms of their main purpose? How so?</li>

 <li>A VAE is generative. Can the same be said of a standard autoencoder? Why or why not?</li>

 <li>Can a VAE be used in place of a standard autoencoder for its purpose you mentioned above?</li>

</ol>

<h2>2.2           Decoder: The Generative Part of the VAE</h2>

In the previous section, we described a general graphical model which also applies to

VAEs. In this section, we will define a more specific generative model that we will use throughout this assignment. This will later be refered to as the <u>decoding </u>part (or decoder) of a VAE. For this assignment we will assume the pixels of our images <em>x</em><em><sub>n </sub></em>in the dataset D are Categorical(<em>p</em>) distributed.

<em>p</em>(<em>z</em><em><sub>n</sub></em>) = N(0<em>,</em><em>I</em><em><sub>D</sub></em>)                                                                               (7)

Cat                                                (8)

where <em>x</em> is the <em>m</em>-th pixel of the <em>n</em>-th image in D, <em>f<sub>θ </sub></em>: R<em><sup>D </sup></em>→ (<em>p</em><sub>1</sub><em>,…,p<sub>k</sub></em>)<em><sup>M </sup></em>is a neural network parameterized by <em>θ </em>that outputs the probabilities of the Categorical distributions for each pixel in <em>x</em><em><sub>n</sub></em>. In other words, <em>p </em>= (<em>p</em><sub>1</sub><em>,…,p<sub>k</sub></em>) are event probabilities of the pixel belonging to value <em>k</em>, where <em>p<sub>i </sub></em>≥ 0 and.

<h3>Question 2.1</h3>

Now that we have defined the model, we can write out an expression for the log probability




of the data D under this model:

<em>N </em>log<em>p</em>(D) = <sup>X </sup>log<em>p</em>(<em>x</em><em><sub>n</sub></em>)

<em>n</em>=1

<em>N </em>Z

X

=         log         <em>p</em>(<em>x</em><em><sub>n</sub></em>|<em>z</em><em><sub>n</sub></em>)<em>p</em>(<em>z</em><em><sub>n</sub></em>)<em>d</em><em>z</em><em><sub>n                                                                     </sub></em>(9)

<em>n</em>=1

<em>N</em>

= <sup>X </sup>logE<em>p</em><sub>(<em>z</em></sub><em><sub>n</sub></em><sub>) </sub>[<em>p</em>(<em>x</em><em><sub>n</sub></em>|<em>z</em><em><sub>n</sub></em>)] <em>n</em>=1




Evaluating log<em>p</em>(<em>x</em><em><sub>n</sub></em>) = logE<em>p</em><sub>(<em>z</em></sub><em><sub>n</sub></em><sub>) </sub>[<em>p</em>(<em>x</em><em><sub>n</sub></em>|<em>z</em><em><sub>n</sub></em>)] involves a very expensive integral. However, Equation 9 hints at a method for approximating it, namely<a href="https://ermongroup.github.io/cs228-notes/inference/sampling/">Monte-Carlo Integration</a><a href="https://ermongroup.github.io/cs228-notes/inference/sampling/">.</a> The log-likelihood can be approximated by drawing samples <em>z</em> from <em>p</em>(<em>z</em><em><sub>n</sub></em>):

log<em>p</em>(<em>x</em><em><sub>n</sub></em>) = logE<em>p</em><sub>(<em>z</em></sub><em><sub>n</sub></em><sub>) </sub>[<em>p</em>(<em>x</em><em><sub>n</sub></em>|<em>z</em><em><sub>n</sub></em>)]                                                                    (10)

(11)

If we increase the number of samples <em>L </em>to infinity, the approximation would be equals to the actual expectation. Hence, the estimator is unbiased and can be used to approximate log<em>p</em>(<em>x</em><em><sub>n</sub></em>) with a sufficient large number of samples.

<h3>Question 2.2</h3>

Although Monte-Carlo Integration with samples from <em>p</em>(<em>z</em><em><sub>n</sub></em>) can be used to approximate log<em>p</em>(<em>x</em><em><sub>n</sub></em>), it is not used for training VAE type of models, because it is inefficient. In a few sentences, describe why it is inefficient and how this efficiency scales with the dimensionality of <em>z</em>. (Hint: you may use Figure 3 in you explanation.)

<h2>2.3           KL Divergence</h2>

Before continuing our discussion about VAEs, we will need to learn about another concept that will help us later: the Kullback-Leibler divergence (KL divergence). It measures how different one probability distribution is from another:

(12)

where <em>q </em>and <em>p </em>are probability distributions in the space of some random variable <em>X</em>.

<h3>Question 2.3</h3>

Assume that <em>q </em>and <em>p </em>in Equation 12, are univariate gaussians:                                   and

. Give two examples of (                                 ): one of which results in a

very small, and one of which has a very large, KL-divergence: <em>D</em><sub>KL</sub>(<em>q</em>||<em>p</em>).

Figure 3. Plot of 2-dimensional latent space and contours of prior and posterior distributions. The red contour shows the prior <em>p</em>(<em>z</em>) which is a Gaussian distribution with zero mean and standard deviation of one. The black points represent samples from the prior <em>p</em>(<em>z</em>). The blue contour shows the posterior distribution <em>p</em>(<em>z</em>|<em>x</em>) for an arbitrary <em>x</em>, which is a complex distribution and here, for example, peaked around (1<em>,</em>1).

In VAEs, we usually set the prior to be a normal distribution with a zero mean and unit variance: <em>p </em>= N(0<em>,</em>1). For this case, we can actually find a closed-form solution of the KL divergence:

(13)

(14)

(15)

(16)

For simplicity, we skipped a few steps in the derivation. You can find the details <a href="https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians">here</a> if you are interested (it is not essential for understanding the VAE). We will need this result for our implementation of the VAE later.

<h2>2.4           The Encoder: <em>q<sub>ϕ</sub></em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>) – Efficiently evaluating the integral</h2>

In the previous section 2.2, we have developed the intuition why we need the posterior <em>p</em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>). Unfortunately, the true posterior <em>p</em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>) is as difficult to compute as <em>p</em>(<em>x</em><em><sub>n</sub></em>) itself. To solve this problem, instead of modeling the true posterior <em>p</em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>), we can learn an <u>approximate posterior distribution</u>, which we refer to as the <u>variational distribution</u>. This variational distribution <em>q</em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>) is used to approximate the (very expensive) posterior <em>p</em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>).

Now we have all the tools to derive an efficient bound on the log-likelihood log<em>p</em>(D). We start from Equation 9 where the log-likelihood objective is written, but for simplicity in notation we write the log-likelihood log<em>p</em>(<em>x</em><em><sub>n</sub></em>) only for a single datapoint.

(multiply by <em>q</em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>)<em>/q</em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>))

(switch expectation distribution)

(Jensen’s inequality)                 (re-arranging)

= E<em>q</em><sub>(<em>z</em></sub><em><sub>n</sub></em><sub>|<em>x</em></sub><em><sub>n</sub></em><sub>) </sub>[log<em>p</em>(<em>x</em><em><sub>n</sub></em>|<em>z</em><em><sub>n</sub></em>)] − <em>KL</em>(<em>q</em>(<em>Z</em>|<em>x</em><em><sub>n</sub></em>)||<em>p</em>(<em>Z</em>))                        (writing 2nd term as KL)

|                                         {z                                        }

Evidence Lower Bound (ELBO)

(17) This is awesome! We have derived a bound on log<em>p</em>(<em>x</em><em><sub>n</sub></em>), exactly the thing we want to optimize, where all terms on the right hand side are computable. Let’s put together what we have derived again in a single line:

log<em>p</em>(<em>x</em><em><sub>n</sub></em>) <sup>≥ </sup>E<em>q</em><sub>(<em>z</em></sub><em><sub>n</sub></em><sub>|<em>x</em></sub><em><sub>n</sub></em><sub>) </sub>[log<em>p</em>(<em>x</em><em><sub>n</sub></em>|<em>z</em><em><sub>n</sub></em>)] − <em>KL</em>(<em>q</em>(<em>Z</em>|<em>x</em><em><sub>n</sub></em>)||<em>p</em>(<em>Z</em>))<em>.</em>

The right side of the equation is referred to as the <em>evidence lowerbound </em>(ELBO) on the log-probability of the data.

This leaves us with the question: How close is the ELBO to log<em>p</em>(<em>x</em><em><sub>n</sub></em>)? With an alternate derivation<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>, we can find the answer. It turns out the gap between log<em>p</em>(<em>x</em><em><sub>n</sub></em>) and the ELBO is exactly <em>KL</em>(<em>q</em>(<em>Z</em>|<em>x</em><em><sub>n</sub></em>)||<em>p</em>(<em>Z</em>|<em>x</em><em><sub>n</sub></em>)) such that:

log<em>p</em>(<em>x</em><em><sub>n</sub></em>)−<em>KL</em>(<em>q</em>(<em>Z</em>|<em>x</em><em><sub>n</sub></em>)||<em>p</em>(<em>Z</em>|<em>x</em><em><sub>n</sub></em>)) = E<em>q</em><sub>(<em>z</em></sub><em><sub>n</sub></em><sub>|<em>x</em></sub><em><sub>n</sub></em><sub>) </sub>[log<em>p</em>(<em>x</em><em><sub>n</sub></em>|<em>z</em><em><sub>n</sub></em>)]−<em>KL</em>(<em>q</em>(<em>Z</em>|<em>x</em><em><sub>n</sub></em>)||<em>p</em>(<em>Z</em><sup>)) </sup>(18)

Now, let’s optimize the ELBO. For this, we define our loss as the mean negative lower bound over samples:

(19)

Note, that we make an explicit distinction between the generative parameters <em>θ </em>and the variational parameters <em>ϕ</em>.

<h3>Question 2.4</h3>

Explain how you can see from Equation 18 that the right hand side has to be a <em>lower bound </em>on the log-probability log<em>p</em>(<em>x</em><em><sub>n</sub></em>)? Why must we optimize the lower-bound,

instead of optimizing the log-probability log<em>p</em>(<em>x</em><em><sub>n</sub></em>) directly?

<h3>Question 2.5</h3>

Now, looking at the two terms on left-hand side of 18: Two things can happen when the lower bound is pushed up. Can you describe what these two things are?

<h2>2.5           Specifying the Encoder <em>q<sub>ϕ</sub></em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>)</h2>

In VAE, we have some freedom to choose the distribution <em>q<sub>ϕ</sub></em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>). In essence, we want to choose something that can closely approximate <em>p</em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>), but we are also free to a select distribution that makes our life easier. We will do exactly that in this case and choose <em>q<sub>ϕ</sub></em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>) to be a factored multivariate normal distribution, i.e.,

<em>q<sub>ϕ</sub></em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>) = N(<em>z</em><em><sub>n</sub></em>|<em>µ<sub>ϕ</sub></em>(<em>x</em><em><sub>n</sub></em>)<em>,</em>diag(Σ<em><sub>ϕ</sub></em>(<em>x</em><em><sub>n</sub></em>)))<em>,                                              </em>(20)

where <em>µ<sub>ϕ </sub></em>: R<em><sup>M </sup></em>→ R<em><sup>D </sup></em>maps an input image to the mean of the multivariate normal over <em>z</em><em><sub>n </sub></em>and  maps the input image to the diagonal of the covariance matrix of that same distribution. Moreover, diag(<em>v</em>) maps a <em>K</em>-dimensional (for any <em>K</em>) input vector <em>v </em>to a <em>K </em>× <em>K </em>matrix such that for <em>i,j </em>∈ {1<em>,…,K</em>}

diag<em> .                                                     </em>(21)

Now we have defined an objective (Equation 19) in terms of an abstract model and variational approximation, we can put everything together using our model definition (Equation 5 and 6) and definition of <em>q<sub>ϕ</sub></em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>) (Equation 20), and we can write down a single objective which we can minimize.

First, we write down the reconstruction term:

Lrecon<em>n          </em>= −E<em>q</em><em>ϕ</em>(<em>z</em>|<em>x</em><em>n</em>)[log<em>p</em><em>θ</em>(<em>x</em><em>n</em>|<em>Z</em>)]

here we used Monte-Carlo integration to approximate the expectation

Cat

Remember that <em>f<sub>θ</sub></em>(·) denotes our decoder. Now let <em>p</em>, then

<em>.</em>

where <em>x</em> if the <em>m</em>-th pixel has the value <em>k</em>, and zero otherwise. In other words, the equation above represents the common cross-entropy loss term. When setting <em>L </em>= 1 (i.e. only one sample for <em>z</em><em><sub>n</sub></em>), we obtain:

where <em>p</em> and <em>z</em><em><sub>n </sub></em>∼ <em>q</em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>). Thus, we can use the cross-entropy loss with respect to the original input <em>x</em><em><sub>n </sub></em>to optimize L<sup>recon</sup><em><sub>n </sub></em>Next, we write down the regularization term:

= <em>D</em><sub>KL</sub>(N(<em>Z</em>|<em>µ<sub>ϕ</sub></em>(<em>x</em><em><sub>n</sub></em>)<em>,</em>diag(Σ<em><sub>ϕ</sub></em>(<em>x</em><em><sub>n</sub></em>))))||N(<em>Z</em>|<strong>0</strong><em>,</em><em>I</em><em><sub>D</sub></em>))

Using the fact that both probability distributions factorize and that the KL-divergence of two factorizable distributions is a sum of KL terms, we can rewrite this to

<em>D</em>

= X<em>D</em>KL(N(<em>Z</em>(<em>d</em>)|<em>µ</em><em>ϕ</em>(<em>x</em><em>n</em>)<em>d</em><em>,</em>Σ<em>ϕ</em>(<em>x</em><em>n</em>)<em>d </em>||N(<em>Z</em>(<em>d</em>)|0<em>,</em>1)) <em>d</em>=1

Let <em>µ<sub>nd </sub></em>= <em>µ<sub>ϕ</sub></em>(<em>x</em><em><sub>n</sub></em>)<em><sub>d </sub></em>and <em>σ<sub>nd </sub></em>= Σ<em><sub>ϕ</sub></em>(<em>x</em><em><sub>n</sub></em>)<em><sub>d</sub></em>, then using the solution we found for question 1.5 we have

<em>.</em>

Hence, we can find the regularization term via the simple equation above.

<h2>2.6           The Reparametrization Trick</h2>

Although we have written down (the terms of) an objective above, we still cannot simply minimize this by taking gradients with regard to <em>θ </em>and <em>ϕ</em>. This is due to the fact that we sample from <em>q<sub>ϕ</sub></em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>) to approximate the E<em>q<sub>ϕ</sub></em><sub>(<em>z</em>|<em>x</em></sub><em><sub>n</sub></em><sub>)</sub>[log<em>p<sub>θ</sub></em>(<em>x</em><em><sub>n</sub></em>|<em>Z</em>)] term. Yet, we need to pass the derivative through these samples if we want to compute the gradient of the encoder parameters, i.e., ∇<em><sub>ϕ</sub></em>L(<em>θ,ϕ</em>). Our posterior approximation <em>q<sub>ϕ</sub></em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>) is parameterized by <em>ϕ</em>. If we want to train <em>q<sub>ϕ</sub></em>(<em>z</em><em><sub>n</sub></em>|<em>x</em><em><sub>n</sub></em>) to maximize the lower bound, and therefore approximate the posterior, we need to have the gradient of the lower-bound with respect to <em>ϕ</em>.

Figure 4. A VAE architecture on MNIST. The encoder distribution <em>q</em>(<em>z</em>|<em>x</em>) maps the input image into latent space. This latent space should follow a unit Gaussian prior <em>p</em>(<em>z</em>). A sample from <em>q</em>(<em>z</em>|<em>x</em>) is used as input to the decoder <em>p</em>(<em>x</em>|<em>z</em>) to reconstruct the image. Figure taken from <a href="https://mlmasteryblog.com/the-rise-and-fall-of-variational-auto-encoders/">this blog</a><a href="https://mlmasteryblog.com/the-rise-and-fall-of-variational-auto-encoders/">.</a> Note that we are using FashionMNIST and not MNIST to train our VAE in this assignment.

<h3>Question 2.7 (3 points)</h3>

Passing the derivative through samples can be done using the <em>reparameterization trick</em>. In a few sentences, explain why the act of sampling usually prevents us from

computing ∇<em><sub>ϕ</sub></em>L, and how the reparameterization trick solves this problem.

<h2>2.7           Putting things together: Building a VAE</h2>

Given everything we have discussed so far, we now have an objective (the <u>evidence lower bound </u>or ELBO) and a way to backpropagate to both <em>θ </em>and <em>ϕ </em>(i.e., the reparametrization trick). Thus, we can now implement a VAE in PyTorch to train on FashionMNIST images.

We will model the encoder <em>q</em>(<em>z</em>|<em>x</em>) and decoder <em>p</em>(<em>x</em>|<em>z</em>) by a deep neural network each, and train them to maximize the data likelihood. See Figure 4 for an overview of the components we need to consider in a VAE.

In the code directory part1, you can find the templates to use for implementing the

VAE. We provide two versions for the training loop: a template in PyTorch Lightning (train_pl.py), and a template in plain PyTorch (train_torch.py). You can choose which you prefer to implement. You only need to implement one of the two training loop templates. If you followed the tutorial notebooks, you might want to give PyTorch Lightning a try as it is less work, more structured and has an automatic saving and logging mechanism. You do not need to be familiar with PyTorch Lightning to the lowest level, but a high-level understanding as from the introduction in <a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html#PyTorch-Lightning">Tutorial</a> <a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html#PyTorch-Lightning">5 </a>is sufficient for implementing the template.

You also need to implement additional functions in utils.py, and the encoder and decoder in the files cnn_encoder_decoder.py. We specified a recommended architecture to start with, but you are allowed to experiment with your own ideas for the models. For the sake of the assignment, it is sufficient to use the recommended architecture to achieve full points. Use the provided unit tests to ensure the correctness of your implementation. Details on the files can be found in the README of part 2.

As a loss objective and test metric, we will use the bits per dimension score (bpd). Bpd is motivated from an information theory perspective and describes how many bits we would need to encode a particular example in our modeled distribution. You can see it as how many bits we would need to store this image on our computer or send it over a network, if we have given our model. The less bits we need, the more likely the example is in our distribution. Hence, we can use bpd as loss metric to minimize. When we test for the bits per dimension on our test dataset, we can judge whether our model generalizes to new samples of the dataset and didn’t in fact memorize the training dataset. In order to calculate the bits per dimension score, we can rely on the negative log-likelihood we got from the ELBO, and change the log base (as bits are binary while NLL is usually exponential):

bpd

where <em>d</em><sub>1</sub><em>,…,d<sub>K </sub></em>are the dimensions of the input excluding any batch dimension. For images, this would be the height, width and channel number. We average over those dimensions in order to have a metric that is comparable across different image resolutions. The nll represents the negative log-likelihood loss L from Equation 19 for a single data point. You should implement this function in utils.py.

<h3>Question 2.8 (12 points)</h3>

Build a Variational Autoencoder in the provided templates, and train it on the FashionMNIST dataset. Both the encoder and decoder should be implemented as a CNN. For the architecture, you can use the same as used in <a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html#Building-the-autoencoder">Tutorial</a> <a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html#Building-the-autoencoder">9</a> about Autoencoders. Note that you have to adjust the output shape of the decoder to 1 × 28 × 28 for FashionMNIST. You can do this by adjusting the output padding of the first transposed convolution in the decoder. Use a latent space size of z_dim=20. Read the provided README to become familiar with the code template.

In your submission, provide a short description (no more than 8 lines) of the used architectures for the encoder and decoder, any hyperparameters and your training steps. Additionally, plot the estimated bit per dimension score of the lower bound on the training and validation set as training progresses, and the final test score. You are allowed to take screenshots of a TensorBoard plot if the axes values are clear.

<u>Note: using the default hyperparameters is sufficient to obtain full points. As a reference, the training loss should start at around 4 bpd, reach below 2.0 after 2 epochs, and end between 1.20-1.25 after 80 epochs.</u>

<h3>Question 2.9 (4 points)</h3>

Plot 64 samples (8 × 8 grid) from your model at three points throughout training

(before training, after training 10 epochs, and after training 80 epochs). You should observe an improvement in the quality of samples. Describe shortly the quality and/or issues of the generated images.

<h3>Question 2.10 (4 points)</h3>

Train a VAE with a 2-dimensional latent space (z_dim=2 in the code). Use this

VAE to plot the data <u>manifold </u>as is done in Figure 4b of [Kingma and Welling, 2014]. This is achieved by taking a two dimensional grid of points in <em>Z</em>-space, and plotting <em>f<sub>θ</sub></em>(<em>Z</em>) = <em>µ</em>|<em>Z</em>. Use the percent point function (ppf, or the inverse CDF) to cover the part of <em>Z</em>-space that has significant density. Implement it in the function visualize_manifold in utils.py, and use a grid size of 20. Are you recognizing any patterns of the positions of the different items of clothing?

<h1>References</h1>

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In Yoshua Bengio and Yann LeCun, editors, <u>3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings</u>, 2015. URL <a href="https://arxiv.org/abs/1409.0473">http</a><a href="https://arxiv.org/abs/1409.0473"><sub>:</sub></a>

<a href="https://arxiv.org/abs/1409.0473">//arxiv.org/abs/1409.0473</a><a href="https://arxiv.org/abs/1409.0473">.</a> 2, 3

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In <u>Advances in neural information processing systems</u>, pages 2672–2680, 2014. 1

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. <u>International Conference on Learning Representations (ICLR)</u>, 2014. 1, 12

Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing flows. In <u>Proceedings of the 32Nd International Conference on International Conference on Machine Learning – Volume 37</u>, ICML’15, pages 1530–1538. JMLR.org, 2015. URL <a href="https://dl.acm.org/citation.cfm?id=3045118.3045281">http://dl.acm.org/citation.cfm?id=3045118.3045281</a><a href="https://dl.acm.org/citation.cfm?id=3045118.3045281">.</a> 1

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In <u>Advances in neural information processing systems</u>, pages 5998–6008, 2017. 3

<a href="#_ftnref1" name="_ftn1">[1]</a> This derivation is not done here, but can be found in for instance Bishop sec 9.4.