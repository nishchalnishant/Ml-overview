---
module: Interview Prep
topic: Ml
subtopic: Probabilistic Graphical Models
status: unread
tags: [interviewprep, ml, ml-probabilistic-graphical-mod]
---
# Probabilistic Graphical Models (PGMs)

---

## What This File Is For

Every topic is structured around the four questions that matter in an interview:
1. What the interviewer is actually testing
2. The reasoning structure — why first-principles thinkers approach it this way
3. The pattern in action — a worked example
4. Common traps — where people go wrong and why

Code examples are retained where they illustrate the concept.

---

## 1. PGM Overview

**What the interviewer is testing:** Whether you understand why graphical models exist — the core problem they solve — and can distinguish directed from undirected models based on the structural implications, not just the naming.

**The reasoning structure:** A joint distribution over $n$ variables in a domain of size $k$ requires $k^n - 1$ parameters — impossible to represent or learn for any real problem. PGMs exploit conditional independence structure encoded as a graph to factorize the joint into smaller, tractable pieces. The graph is a set of encoded assumptions: edges that are absent represent declared independence.

**Directed (Bayesian Networks / DAG):**
$$P(X_1, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid Pa(X_i))$$
Each node needs only a conditional probability table (CPT) over its parents. The direction encodes a generative story — this variable is caused by these parents.

**Undirected (Markov Random Fields / Gibbs distribution):**
$$P(X_1, \ldots, X_n) = \frac{1}{Z} \prod_{c \in \text{Cliques}} \phi_c(X_c)$$
Clique potentials $\phi_c$ express soft preferences over joint assignments. No causal direction. The partition function $Z$ normalizes — and is the root of intractability.

| Property | Bayesian Networks (DAG) | Markov Random Fields (undirected) |
|---|---|---|
| Edge semantics | Causal / generative direction | Symmetric affinity / compatibility |
| Factorization | Chain rule over CPTs | Clique potentials |
| Normalization | Always normalized | Requires partition function Z |
| Cycles | Forbidden (acyclic) | Allowed |
| Typical use | Generative models, causal reasoning | Spatial models, discriminative tasks |

**D-separation and conditional independence:** D-separation determines when a set of nodes $Z$ blocks all paths between $X$ and $Y$, rendering $X \perp Y \mid Z$.

| Pattern | Structure | Blocked by Z? |
|---|---|---|
| Chain | $X \to Z \to Y$ | Yes — Z observed |
| Fork | $X \leftarrow Z \to Y$ | Yes — Z observed |
| Collider (v-structure) | $X \to Z \leftarrow Y$ | No — Z unobserved blocks; Z observed opens |

The collider is the counterintuitive case: observing the common effect of two causes makes those causes dependent (explaining away). This is Berkson's paradox.

**The pattern in action:** A spam filter models $P(\text{spam} \mid \text{words})$. Naive Bayes is a BN: Class $\to$ Feature$_1$, Class $\to$ Feature$_2$, etc. The directed edges encode the generative assumption that spam status causes the words. For image segmentation, pixel labels have symmetric spatial compatibility — no causal direction — so an MRF is more natural, encoding that neighboring pixels should have similar labels.

**Common traps:**
- Saying MRFs require specifying causal direction. They do not — the absence of direction is the point.
- Forgetting that the partition function $Z$ is the fundamental obstacle. Computing $Z$ requires summing over all variable configurations — exponential in the number of variables. This is why approximate inference exists.

---

## 2. Bayesian Networks

**What the interviewer is testing:** Whether you can reason about conditional independence in a BN, perform conceptual variable elimination, and connect BNs to practical models (Naive Bayes, causal inference).

**The reasoning structure:** A BN encodes a factored joint distribution. The key skill is reading conditional independencies from graph structure. The Markov blanket of a node is its parents, children, and co-parents — it is conditionally independent of everything else given its Markov blanket.

**Alarm Network example:**
```
Burglary  Earthquake
    \       /
     v     v
      Alarm
     /     \
    v       v
JohnCalls MaryCalls
```
```
P(B,E,A,J,M) = P(B) * P(E) * P(A|B,E) * P(J|A) * P(M|A)
```
The directed structure means Burglary and Earthquake are marginally independent ($B \perp E$) but become dependent given Alarm ($B \not\perp E \mid A$) — the collider pattern.

**Naive Bayes** is a BN with Class $\to$ Feature$_i$ for all $i$:
```
P(C, F1, ..., Fn) = P(C) * prod_i P(Fi | C)
```
The conditional independence assumption (features independent given class) is often violated in practice but the classifier is surprisingly robust.

**Variable elimination for exact inference:**
1. Initialize factors from all CPTs, instantiated with evidence
2. Choose an elimination ordering over hidden variables
3. For each variable $Z$: multiply all factors containing $Z$, sum out $Z$
4. Multiply remaining factors; normalize

Complexity is exponential in the treewidth of the graph. For polytrees (singly connected), inference is polynomial. For general graphs, exact inference is #P-hard.

**The pattern in action:** In the alarm network, query $P(J = \text{true} \mid B = \text{true})$: instantiate $B = \text{true}$, eliminate $E$, then $A$, then $M$. At each step you multiply factors and sum out one variable. The result is a factor over $J$ which you normalize. The efficiency comes from eliminating variables in a good ordering — never creating large intermediate factors.

**Common traps:**
- Assuming good elimination orderings are easy to find. Finding the optimal elimination ordering (minimizing treewidth) is NP-hard. Heuristics like min-fill are used in practice.
- Confusing the Markov blanket with the neighborhood. The Markov blanket includes co-parents (other parents of the node's children) — without them, the node is not screened off from its children's co-causes.

---

## 3. Hidden Markov Models

**What the interviewer is testing:** Whether you can map the three core HMM problems to the three core algorithms, explain the DP structure of each, and distinguish HMM from CRF.

**The reasoning structure:** An HMM models a sequence of observations generated by a latent Markov chain:

- **States:** $S = \{s_1, \ldots, s_N\}$ (hidden)
- **Observations:** $O = \{o_1, \ldots, o_K\}$ (visible)
- **Initial distribution:** $\pi_i = P(q_1 = s_i)$
- **Transition matrix:** $A[i,j] = P(q_{t+1} = s_j \mid q_t = s_i)$
- **Emission matrix:** $B[i,k] = P(o_t = o_k \mid q_t = s_i)$

```
q1 -> q2 -> q3 -> ... -> qT
|     |     |            |
v     v     v            v
o1    o2    o3  ...      oT
```

Three problems, three algorithms:

**Problem 1 — Evaluation (Forward Algorithm):** Compute $P(O \mid \lambda)$.
Define $\alpha_t(i) = P(o_1, \ldots, o_t, q_t = s_i \mid \lambda)$:
```
alpha_1(i) = pi_i * B[i, o1]
alpha_{t+1}(j) = B[j, o_{t+1}] * sum_i [ alpha_t(i) * A[i,j] ]
P(O | lambda) = sum_i alpha_T(i)
```
Complexity: $O(N^2 T)$ vs $O(N^T)$ for brute force.

**Problem 2 — Decoding (Viterbi):** Find most likely state sequence $q^* = \arg\max_q P(q \mid O, \lambda)$.
Same DP structure as forward but replace sum with max and store backpointers:
```
delta_1(i) = pi_i * B[i, o1]
delta_{t+1}(j) = B[j, o_{t+1}] * max_i [ delta_t(i) * A[i,j] ]
psi_{t+1}(j) = argmax_i [ delta_t(i) * A[i,j] ]
```
Backtrack through $\psi$ to recover the optimal sequence.

**Problem 3 — Learning (Baum-Welch / EM):** Estimate model parameters $\lambda$.
E-step: compute $\gamma_t(i) = P(q_t = s_i \mid O, \lambda)$ and $\xi_t(i,j) = P(q_t = s_i, q_{t+1} = s_j \mid O, \lambda)$.
M-step: re-estimate $\pi$, $A$, $B$ from expected counts. Guaranteed to find a local maximum of likelihood.

### Python Code — HMM with hmmlearn

```python
import numpy as np
from hmmlearn import hmm

np.random.seed(42)

true_model = hmm.GaussianHMM(n_components=2, covariance_type="full")
true_model.startprob_ = np.array([0.6, 0.4])
true_model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])
true_model.means_ = np.array([[0.0], [5.0]])
true_model.covars_ = np.array([[[1.0]], [[1.0]]])

X, Z_true = true_model.sample(200)

model = hmm.GaussianHMM(n_components=2, covariance_type="full",
                         n_iter=100, random_state=42)
model.fit(X)

log_prob = model.score(X)
log_prob_viterbi, Z_pred = model.decode(X, algorithm="viterbi")

from sklearn.metrics import accuracy_score
acc1 = accuracy_score(Z_true, Z_pred)
acc2 = accuracy_score(Z_true, 1 - Z_pred)
print(f"Viterbi decoding accuracy: {max(acc1, acc2):.3f}")
```

**The pattern in action:** Speech recognition uses HMMs where hidden states correspond to phoneme segments and observations are acoustic feature vectors. The forward algorithm computes the likelihood of the acoustic sequence under a given word's phoneme model. Viterbi finds the most likely phoneme sequence. Baum-Welch estimates transition and emission parameters from labeled speech data.

**Common traps:**
- Forgetting to implement in log space. Forward/Viterbi probabilities underflow to zero for long sequences — always use log probabilities with log-sum-exp.
- Confusing the three algorithms. Forward: sum-product, returns a scalar probability. Viterbi: max-product with backpointers, returns a sequence. Baum-Welch: EM with forward-backward in the E-step.

---

## 4. Markov Random Fields

**What the interviewer is testing:** Whether you understand that MRFs represent undirected dependencies through clique potentials, why the partition function is the fundamental obstacle, and how specific models (RBMs) exploit structure to make inference tractable.

**The reasoning structure:** An MRF's distribution factorizes over clique potentials:
$$P(X) = \frac{1}{Z} \prod_{c \in C} \phi_c(X_c)$$

The partition function $Z = \sum_X \prod_c \phi_c(X_c)$ normalizes — and computing it requires summing over all configurations: exponential in the number of variables. The Hammersley-Clifford theorem establishes the equivalence between Markov properties of the graph and Gibbs factorization.

**Image segmentation (grid MRF):**
```
P(Y | X) ∝  prod_i phi_data(Yi, Xi)  *  prod_{(i,j) in Edges} phi_smooth(Yi, Yj)
```
- `phi_data`: pixel intensity matches label distribution
- `phi_smooth = exp(-lambda * 1[Yi != Yj])`: penalty for neighboring pixels with different labels

**Boltzmann Machine:** An MRF over binary units with pairwise potentials:
$$P(v, h) = \frac{1}{Z} \exp(v^T W h + b^T v + c^T h)$$

**Restricted Boltzmann Machine (RBM):** No intra-layer connections. The bipartite structure makes conditional distributions tractable:
```
P(h | v) = prod_j P(hj | v)     # independent given visible
P(v | h) = prod_i P(vi | h)     # independent given hidden
```
This enables block Gibbs sampling (contrastive divergence) for training.

**The pattern in action:** A content-based image segmentation task. You model each pixel's label as depending on its observed intensity (data term) and its neighbors' labels (smoothness term). The MRF encodes the assumption that adjacent pixels tend to have the same label. Inference is via loopy belief propagation or graph cuts — both approximate methods for dealing with the intractable partition function.

**Common traps:**
- Assuming the partition function is computable. For most real-world MRFs it is not — this is why all training and inference algorithms for MRFs involve approximations.
- Confusing RBMs with general Boltzmann machines. RBMs have the restricted bipartite structure that makes conditionals tractable. General Boltzmann machines require MCMC even for conditionals.

---

## 5. Belief Propagation

**What the interviewer is testing:** Whether you understand when belief propagation is exact (trees), when it is approximate (loopy graphs), and why it works as well as it does despite no convergence guarantee for loopy graphs.

**The reasoning structure:** Belief propagation passes messages along edges. On a tree, a node's marginal is determined by the messages from its neighbors — there are no cycles, so each path of information is unique.

**Sum-product messages (factor graphs, trees):**
```
Variable to factor:    mu_{x -> f}(x)  = prod_{g in N(x) \ f} mu_{g -> x}(x)
Factor to variable:    mu_{f -> x}(x)  = sum_{~x} [ f(X_f) * prod_{y in N(f) \ x} mu_{y -> f}(y) ]
```

After convergence on a tree, the marginal at node $i$:
$$P(x_i) \propto \phi_i(x_i) \prod_{k \in N(i)} \mu_{k \to i}(x_i)$$

Complexity: $O(N \cdot K^2)$ for $N$ nodes with $K$ states.

**Loopy BP:** Run the same equations on graphs with cycles, iterating until convergence or max iterations. Not guaranteed to converge; when it does, the fixed-point marginals are approximations. Equivalent to the Bethe free energy approximation. Works well empirically in LDPC codes and computer vision despite the lack of theoretical guarantees.

**Max-product:** Replace sum with max in messages to compute MAP assignments instead of marginals. On trees, max-product equals Viterbi. On loopy graphs, approximate MAP decoding.

**Scheduling strategies:** Synchronous (all messages updated simultaneously), asynchronous (sequential updates), residual BP (update highest-residual messages first — faster convergence).

**The pattern in action:** In error-correcting codes (LDPC), the factor graph has cycles but belief propagation converges quickly and delivers near-optimal decoding. The reason: the cycles are long, so the short loops of information that violate the tree assumption are rare relative to the information flowing along long paths. Loopy BP performs poorly on graphs with many short cycles.

**Common traps:**
- Applying BP to dense graphs and expecting convergence. BP converges reliably on sparse graphs with long cycles; it frequently fails on dense graphs.
- Confusing sum-product (computes marginals) with max-product (computes MAP). They use the same message structure but different aggregation operations.

---

## 6. Variational Inference

**What the interviewer is testing:** Whether you can derive the ELBO from first principles, explain what mean-field assumes and why it leads to biased posteriors, and connect variational inference to VAEs.

**The reasoning structure:** The exact posterior $P(Z \mid X)$ is intractable for most models. Variational inference introduces a tractable family $q(Z)$ and minimizes $D_{KL}(q(Z) \| P(Z \mid X))$. Since the KL is also intractable, the equivalent objective is maximizing the ELBO.

**ELBO derivation:**
$$\log P(X) = \log \int P(X, Z) \, dZ$$
$$= \log \int q(Z) \cdot \frac{P(X, Z)}{q(Z)} \, dZ$$
$$\geq \int q(Z) \log \frac{P(X, Z)}{q(Z)} \, dZ \quad \text{(Jensen's inequality)}$$
$$= \mathbb{E}_q[\log P(X, Z)] - \mathbb{E}_q[\log q(Z)]$$
$$= \mathbb{E}_q[\log P(X, Z)] + H[q] = \text{ELBO}(q)$$

The gap is $\log P(X) - \text{ELBO}(q) = D_{KL}(q(Z) \| P(Z \mid X)) \geq 0$.

**Mean-field approximation:** Assume $q(Z) = \prod_i q_i(Z_i)$ — all latents are independent under $q$. Each factor's optimal update:
$$\log q_j^*(Z_j) = \mathbb{E}_{q_{-j}}[\log P(X, Z)] + \text{const}$$

**CAVI (Coordinate Ascent VI):** Cycle through factors, updating each to its optimal closed form, until ELBO converges.

**Why mean-field underestimates posterior variance:** Mean-field minimizes $D_{KL}(q \| P(Z \mid X))$. This is the reverse KL, which is zero-forcing — $q$ concentrates on modes of $P$ and avoids regions where $P$ is small. The result is an approximate posterior that is too narrow and too confident.

**VAE connection:**
$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$
Encoder $q_\phi$ is the variational distribution (amortized — shared across all $x$). Decoder $p_\theta$ is the generative model. The reparameterization trick $z = \mu + \sigma \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$ allows gradients to flow through sampling.

**The pattern in action:** LDA inference is a textbook CAVI application. The latent variables are topic assignments $z$ and topic distributions $\theta$. Mean-field VI factorizes over all $z_{d,n}$ and $\theta_d$ independently, allowing closed-form Dirichlet-Categorical updates. The approximation is biased (true posterior has correlations between topic assignments and topic proportions that mean-field ignores), but it is fast and scales to millions of documents.

**Common traps:**
- Saying VI is "just minimizing KL divergence." Minimizing $D_{KL}(q \| P)$ is the right framing, but the direction matters: reverse KL produces mode-seeking behavior; forward KL would produce mean-seeking behavior. Most VI uses reverse KL.
- Confusing ELBO maximization with evidence maximization. Maximizing ELBO maximizes a lower bound on $\log P(X)$, not $\log P(X)$ itself. The gap is the KL from $q$ to the true posterior.

---

## 7. MCMC Methods

**What the interviewer is testing:** Whether you can explain why MCMC produces correct samples (detailed balance) and compare Gibbs, MH, and HMC on the correct criteria (computational cost, mixing speed, applicability).

**The reasoning structure:** MCMC constructs a Markov chain whose stationary distribution is the target distribution $\pi(x)$. By drawing samples from a long-running chain, you obtain approximate samples from $\pi$ — asymptotically exact, unlike variational inference.

**Gibbs Sampling:** Sample each variable from its full conditional, holding all others fixed:
$$X_i^{(t+1)} \sim P(X_i \mid X_{-i}^{(t)})$$
A special case of MH with acceptance rate 1. Requires tractable full conditionals — available in conjugate models.

```python
def gibbs_sampling_ising(J, n_steps=1000, beta=1.0):
    N = len(J) + 1
    x = np.random.choice([-1, 1], size=N)
    samples = []
    for _ in range(n_steps):
        for i in range(N):
            h = 0
            if i > 0: h += J[i-1] * x[i-1]
            if i < N - 1: h += J[i] * x[i+1]
            p_plus = 1.0 / (1.0 + np.exp(-2 * beta * h))
            x[i] = 1 if np.random.rand() < p_plus else -1
        samples.append(x.copy())
    return np.array(samples)
```

**Metropolis-Hastings:** General MCMC for any target:
```
1. Propose x' ~ q(x' | x^(t))
2. Compute acceptance ratio:
   alpha = min(1, pi(x') * q(x | x') / (pi(x^(t)) * q(x' | x)))
3. Accept x' with probability alpha; else stay at x^(t)
```
Only $\pi$ up to a normalizing constant is needed — the normalization cancels in the ratio. This is the key property that makes MCMC useful when $Z$ is intractable.

**Hamiltonian Monte Carlo (HMC):** Uses gradient information to make long-distance moves:
$$H(x, p) = U(x) + K(p), \quad U(x) = -\log \pi(x), \quad K(p) = \frac{p^T M^{-1} p}{2}$$
Leapfrog integration for $L$ steps of size $\epsilon$; MH accept/reject on $H$. Mixes far faster than random-walk MH in high dimensions because it follows the local geometry.

**NUTS (No-U-Turn Sampler):** Automates HMC by adaptively choosing $L$, eliminating the need to tune the number of leapfrog steps. Standard in Stan, PyMC.

| Method | Bias | Variance | Scalability | Use when |
|---|---|---|---|---|
| Mean-field VI | Biased | Low | High | Large scale, conjugate models |
| Gibbs | Asymptotically unbiased | High | Medium | Conjugate models, sparse graphs |
| MH | Asymptotically unbiased | High | Low | Any target, easy to implement |
| HMC / NUTS | Asymptotically unbiased | Lower | Lower | Continuous latents, medium scale |

**Convergence diagnostics:**

| Diagnostic | Description | Target |
|---|---|---|
| R-hat (Gelman-Rubin) | Between-chain vs within-chain variance ratio | < 1.01 |
| ESS (Effective Sample Size) | $N / (1 + 2\sum_k \rho_k)$ | > 100 per chain |
| Trace plots | Visual inspection of mixing | Stationary, well-mixed |

**The pattern in action:** A Bayesian hierarchical model for multi-site clinical trial analysis has a posterior over hundreds of continuous latent parameters (site effects, treatment effects, variance components). Gibbs requires conjugate conditionals — not available here. MH with random-walk proposals mixes slowly in 200 dimensions. HMC/NUTS with gradient-informed proposals mixes efficiently and is the standard choice in Stan.

**Common traps:**
- R-hat < 1.01 is necessary but not sufficient for convergence — also check ESS and trace plots.
- Discarding too few warmup samples. The chain must reach stationarity before samples are useful.
- Using MCMC without gradient when HMC is available. For continuous targets, HMC dramatically outperforms random-walk methods.

---

## 8. Latent Dirichlet Allocation

**What the interviewer is testing:** Whether you can describe the generative model precisely, explain why Dirichlet priors are used, and distinguish the inference choices (Gibbs vs variational) and their tradeoffs.

**The reasoning structure:** LDA models each document as a mixture of topics, where each topic is a distribution over vocabulary words.

**Generative process:**
```
For each topic k=1,...,K:    phi_k ~ Dirichlet(beta)        [topic-word distribution]
For each document d=1,...,D:
    theta_d ~ Dirichlet(alpha)                              [document-topic proportion]
    For each word n=1,...,Nd:
        z_{d,n} ~ Categorical(theta_d)                      [topic assignment]
        w_{d,n} ~ Categorical(phi_{z_{d,n}})                [word drawn from topic]
```

Dirichlet is the conjugate prior to Categorical — this enables closed-form posterior updates. $\alpha$ controls document-topic sparsity (small $\alpha$ → documents focus on few topics). $\beta$ controls topic-word sparsity (small $\beta$ → topics focus on few words).

**Collapsed Gibbs sampling:** Integrate out $\theta$ and $\phi$ analytically (Dirichlet-Categorical conjugacy), then sample topic assignments $z$:
$$P(z_{d,n} = k \mid z_{-d,n}, W) \propto (n_{d,k}^{-d,n} + \alpha)(n_{k,w}^{-d,n} + \beta) / (n_k^{-d,n} + V\beta)$$
Only integer counts need to be tracked — fast and memory-efficient.

### Python Code — LDA with sklearn

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

categories = ['sci.space', 'rec.sport.hockey', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='train', categories=categories,
                                 remove=('headers', 'footers', 'quotes'))

vectorizer = CountVectorizer(max_df=0.95, min_df=5, max_features=2000,
                              stop_words='english')
X_counts = vectorizer.fit_transform(newsgroups.data)
vocab = vectorizer.get_feature_names_out()

n_topics = 3
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=20,
                                  learning_method='online', random_state=42,
                                  doc_topic_prior=0.1, topic_word_prior=0.01)
lda.fit(X_counts)
print(f"Perplexity: {lda.perplexity(X_counts):.1f}")

n_top_words = 10
for topic_idx, topic in enumerate(lda.components_):
    top_idx = topic.argsort()[-n_top_words:][::-1]
    top_words = [vocab[i] for i in top_idx]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")
```

**The pattern in action:** A corpus of scientific papers — LDA discovers topics corresponding to "machine learning" (gradient, model, training), "biology" (gene, protein, cell), and "physics" (quantum, particle, energy) without supervision. The number of topics $K$ is a hyperparameter: too small and distinct topics merge; too large and topics fragment into redundant variations. Perplexity on held-out data and coherence scores (PMI of top words) guide $K$ selection.

**Common traps:**
- Topic indices are not identifiable — topics may be permuted across different runs. Never compare topic $k=3$ from two independent runs.
- sklearn's LDA uses online variational EM, not collapsed Gibbs — it is fast but produces biased estimates. For research-quality topic models, use collapsed Gibbs (gensim).
- Forgetting that LDA assumes a bag-of-words representation — word order is discarded.

---

## 9. Conditional Random Fields

**What the interviewer is testing:** Whether you can explain why CRFs outperform HMMs for structured prediction, and what the discriminative-generative distinction means for feature design.

**The reasoning structure:** An HMM is a generative model — it models the joint $P(X, Y)$ and requires modeling the emission distribution $P(X \mid Y)$. This forces independence assumptions on features: each observation only depends on the current state. A CRF models $P(Y \mid X)$ directly, sidestepping the need to model the observation distribution and allowing arbitrary overlapping features of the entire observation sequence.

**Linear-chain CRF for sequence labeling:**
$$P(Y \mid X) = \frac{1}{Z(X)} \exp\left(\sum_{t=1}^{T} \sum_k \lambda_k f_k(y_{t-1}, y_t, X, t)\right)$$

Feature functions $f_k$ can depend on the entire sequence $X$ — you can use a future context word, the capitalization pattern of any word, or the presence of a word in a gazetteer. HMMs cannot use such features without violating the emission independence assumption.

**Inference:** Forward-backward for marginals (used in training gradient computation). Viterbi for MAP decoding (same recurrence as HMM Viterbi but with CRF potentials).

The partition function $Z(X)$ conditions on $X$, so it must be recomputed per sequence — unlike MRFs where $Z$ is a global constant.

### Python Code — CRF with sklearn-crfsuite

```python
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

def word_features(sentence, i):
    word = sentence[i][0]
    pos  = sentence[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'pos': pos,
    }
    if i > 0:
        pw, pp = sentence[i-1][0], sentence[i-1][1]
        features.update({'-1:word.lower()': pw.lower(), '-1:pos': pp})
    else:
        features['BOS'] = True
    if i < len(sentence) - 1:
        nw, np_ = sentence[i+1][0], sentence[i+1][1]
        features.update({'+1:word.lower()': nw.lower(), '+1:pos': np_})
    else:
        features['EOS'] = True
    return features

def sentence_to_features(s): return [word_features(s, i) for i in range(len(s))]
def sentence_to_labels(s): return [label for _, _, label in s]

templates = [
    [("John", "NNP", "B-PER"), ("works", "VBZ", "O"), ("at", "IN", "O"),
     ("Google", "NNP", "B-ORG"), ("in", "IN", "O"), ("London", "NNP", "B-LOC")],
    [("Apple", "NNP", "B-ORG"), ("CEO", "NN", "O"), ("Tim", "NNP", "B-PER"),
     ("Cook", "NNP", "I-PER"), ("visited", "VBD", "O"), ("Paris", "NNP", "B-LOC")],
]

import random; random.seed(42)
sentences = templates * 50
random.shuffle(sentences)

X_all = [sentence_to_features(s) for s in sentences]
y_all = [sentence_to_labels(s) for s in sentences]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2)

crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1,
                             max_iterations=100, all_possible_transitions=True)
crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)
labels = [l for l in crf.classes_ if l != 'O']
print(metrics.flat_classification_report(y_test, y_pred, labels=labels))
```

**The pattern in action:** Named entity recognition (NER). An HMM must model how each word is emitted from each entity type — requiring a probability table $P(\text{word} \mid \text{entity type})$ across a 100k-word vocabulary. A CRF avoids this: it uses features like "is the word capitalized?", "is it in a person name lexicon?", "does it follow a title word?" — features that fire across the entire context window. The CRF is trained discriminatively to distinguish entity sequences, not generatively to produce sentences.

**Common traps:**
- Treating CRF and HMM as equivalent with different parameter estimation. The key distinction is the feature expressiveness — CRFs can use overlapping, non-independent features; HMMs fundamentally cannot.
- Forgetting that $Z(X)$ must be recomputed for each sequence in a CRF. This is more expensive than a global $Z$ but is what enables the discriminative conditioning.

---

## Quick Diagnostics

**If asked about the difference between directed and undirected models:**
Directed (BN) encodes causal/generative direction and is always normalized. Undirected (MRF) encodes symmetric affinity and requires the partition function $Z$ for normalization — which is typically intractable. This intractability drives the entire field of approximate inference.

**If asked about VI vs MCMC:**
VI is faster and scales better; MCMC is asymptotically unbiased. VI uses reverse KL, which is mode-seeking and underestimates posterior variance. MCMC produces correct posterior samples given enough time. For large-scale applications, VI is standard; for accurate posterior quantification in smaller models, HMC/NUTS is preferred.

**If asked about HMM vs CRF:**
HMM is generative — models $P(X, Y)$, requires emission distribution, features must be conditionally independent given state. CRF is discriminative — models $P(Y \mid X)$, no emission assumption, features can be arbitrary functions of the entire observation sequence. Use CRF when rich feature engineering is possible and the generative model's assumptions are violated.

## Rapid Recall

### Saying MRFs require specifying causal direction. They do not
- Direct Answer: the absence of direction is the point.
- Why: This matters because it tells you how to reason about saying mrfs require specifying causal direction. they do not.
- Pitfall: Don't answer "Saying MRFs require specifying causal direction. They do not" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: the absence of direction is the point.

### Forgetting that the partition function $Z$ is the fundamental obstacle. Computing $Z$ requires summing over all variable configurations
- Direct Answer: exponential in the number of variables. This is why approximate inference exists.
- Why: This matters because it tells you how to reason about forgetting that the partition function $z$ is the fundamental obstacle. computing $z$ requires summing over all variable configurations.
- Pitfall: Don't answer "Forgetting that the partition function $Z$ is the fundamental obstacle. Computing $Z$ requires summing over all variable configurations" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: exponential in the number of variables. This is why approximate inference exists.

### Assuming good elimination orderings are easy to find. Finding the optimal elimination ordering (minimizing treewidth) is NP-hard. Heuristics like min-fill are used in practice.
- Direct Answer: Assuming good elimination orderings are easy to find. Finding the optimal elimination ordering (minimizing treewidth) is NP-hard. Heuristics like min-fill are used in practice.
- Why: This matters because it tells you how to reason about assuming good elimination orderings are easy to find. finding the optimal elimination ordering (minimizing treewidth) is np-hard. heuristics like min-fill are used in practice..
- Pitfall: Don't answer "Assuming good elimination orderings are easy to find. Finding the optimal elimination ordering (minimizing treewidth) is NP-hard. Heuristics like min-fill are used in practice." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Assuming good elimination orderings are easy to find. Finding the optimal elimination ordering (minimizing treewidth) is NP-hard. Heuristics like min-fill are used in practice.

### Confusing the Markov blanket with the neighborhood. The Markov blanket includes co-parents (other parents of the node's children)
- Direct Answer: without them, the node is not screened off from its children's co-causes.
- Why: This matters because it tells you how to reason about confusing the markov blanket with the neighborhood. the markov blanket includes co-parents (other parents of the node's children).
- Pitfall: Don't answer "Confusing the Markov blanket with the neighborhood. The Markov blanket includes co-parents (other parents of the node's children)" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: without them, the node is not screened off from its children's co-causes.

### States
- Direct Answer: $S = \{s_1, \ldots, s_N\}$ (hidden)
- Why: This matters because it tells you how to reason about states.
- Pitfall: Don't answer "States" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: $S = \{s_1, \ldots, s_N\}$ (hidden)

### Observations
- Direct Answer: $O = \{o_1, \ldots, o_K\}$ (visible)
- Why: This matters because it tells you how to reason about observations.
- Pitfall: Don't answer "Observations" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: $O = \{o_1, \ldots, o_K\}$ (visible)

### Initial distribution
- Direct Answer: $\pi_i = P(q_1 = s_i)$
- Why: This matters because it tells you how to reason about initial distribution.
- Pitfall: Don't answer "Initial distribution" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: $\pi_i = P(q_1 = s_i)$

### Transition matrix
- Direct Answer: $A[i,j] = P(q_{t+1} = s_j \mid q_t = s_i)$
- Why: This matters because it tells you how to reason about transition matrix.
- Pitfall: Don't answer "Transition matrix" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: $A[i,j] = P(q_{t+1} = s_j \mid q_t = s_i)$

### Emission matrix
- Direct Answer: $B[i,k] = P(o_t = o_k \mid q_t = s_i)$
- Why: This matters because it tells you how to reason about emission matrix.
- Pitfall: Don't answer "Emission matrix" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: $B[i,k] = P(o_t = o_k \mid q_t = s_i)$

### Forgetting to implement in log space. Forward/Viterbi probabilities underflow to zero for long sequences
- Direct Answer: always use log probabilities with log-sum-exp.
- Why: This matters because it tells you how to reason about forgetting to implement in log space. forward/viterbi probabilities underflow to zero for long sequences.
- Pitfall: Don't answer "Forgetting to implement in log space. Forward/Viterbi probabilities underflow to zero for long sequences" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: always use log probabilities with log-sum-exp.

### Confusing the three algorithms. Forward
- Direct Answer: sum-product, returns a scalar probability. Viterbi: max-product with backpointers, returns a sequence. Baum-Welch: EM with forward-backward in the E-step.
- Why: This matters because it tells you how to reason about confusing the three algorithms. forward.
- Pitfall: Don't answer "Confusing the three algorithms. Forward" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: sum-product, returns a scalar probability. Viterbi: max-product with backpointers, returns a sequence. Baum-Welch: EM with forward-backward in the E-step.

### phi_data
- Direct Answer: pixel intensity matches label distribution
- Why: This matters because it tells you how to reason about phi_data.
- Pitfall: Don't answer "phi_data" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: pixel intensity matches label distribution

### phi_smooth = exp(-lambda * 1[Yi != Yj])
- Direct Answer: penalty for neighboring pixels with different labels
- Why: This matters because it tells you how to reason about phi_smooth = exp(-lambda * 1[yi != yj]).
- Pitfall: Don't answer "phi_smooth = exp(-lambda * 1[Yi != Yj])" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: penalty for neighboring pixels with different labels

### Assuming the partition function is computable. For most real-world MRFs it is not
- Direct Answer: this is why all training and inference algorithms for MRFs involve approximations.
- Why: This matters because it tells you how to reason about assuming the partition function is computable. for most real-world mrfs it is not.
- Pitfall: Don't answer "Assuming the partition function is computable. For most real-world MRFs it is not" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: this is why all training and inference algorithms for MRFs involve approximations.

### Confusing RBMs with general Boltzmann machines. RBMs have the restricted bipartite structure that makes conditionals tractable. General Boltzmann machines require MCMC even for conditionals.
- Direct Answer: Confusing RBMs with general Boltzmann machines. RBMs have the restricted bipartite structure that makes conditionals tractable. General Boltzmann machines require MCMC even for conditionals.
- Why: This matters because it tells you how to reason about confusing rbms with general boltzmann machines. rbms have the restricted bipartite structure that makes conditionals tractable. general boltzmann machines require mcmc even for conditionals..
- Pitfall: Don't answer "Confusing RBMs with general Boltzmann machines. RBMs have the restricted bipartite structure that makes conditionals tractable. General Boltzmann machines require MCMC even for conditionals." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Confusing RBMs with general Boltzmann machines. RBMs have the restricted bipartite structure that makes conditionals tractable. General Boltzmann machines require MCMC even for co…

### Applying BP to dense graphs and expecting convergence. BP converges reliably on sparse graphs with long cycles; it frequently fails on dense graphs.
- Direct Answer: Applying BP to dense graphs and expecting convergence. BP converges reliably on sparse graphs with long cycles; it frequently fails on dense graphs.
- Why: This matters because it tells you how to reason about applying bp to dense graphs and expecting convergence. bp converges reliably on sparse graphs with long cycles; it frequently fails on dense graphs..
- Pitfall: Don't answer "Applying BP to dense graphs and expecting convergence. BP converges reliably on sparse graphs with long cycles; it frequently fails on dense graphs." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Applying BP to dense graphs and expecting convergence. BP converges reliably on sparse graphs with long cycles; it frequently fails on dense graphs.

### Confusing sum-product (computes marginals) with max-product (computes MAP). They use the same message structure but different aggregation operations.
- Direct Answer: Confusing sum-product (computes marginals) with max-product (computes MAP). They use the same message structure but different aggregation operations.
- Why: This matters because it tells you how to reason about confusing sum-product (computes marginals) with max-product (computes map). they use the same message structure but different aggregation operations..
- Pitfall: Don't answer "Confusing sum-product (computes marginals) with max-product (computes MAP). They use the same message structure but different aggregation operations." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Confusing sum-product (computes marginals) with max-product (computes MAP). They use the same message structure but different aggregation operations.

### Saying VI is "just minimizing KL divergence." Minimizing $D_{KL}(q \| P)$ is the right framing, but the direction matters
- Direct Answer: reverse KL produces mode-seeking behavior; forward KL would produce mean-seeking behavior. Most VI uses reverse KL.
- Why: This matters because it tells you how to reason about saying vi is "just minimizing kl divergence." minimizing $d_{kl}(q \| p)$ is the right framing, but the direction matters.
- Pitfall: Don't answer "Saying VI is "just minimizing KL divergence." Minimizing $D_{KL}(q \| P)$ is the right framing, but the direction matters" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: reverse KL produces mode-seeking behavior; forward KL would produce mean-seeking behavior. Most VI uses reverse KL.

### Confusing ELBO maximization with evidence maximization. Maximizing ELBO maximizes a lower bound on $\log P(X)$, not $\log P(X)$ itself. The gap is the KL from $q$ to the true posterior.
- Direct Answer: Confusing ELBO maximization with evidence maximization. Maximizing ELBO maximizes a lower bound on $\log P(X)$, not $\log P(X)$ itself. The gap is the KL from $q$ to the true posterior.
- Why: This matters because it tells you how to reason about confusing elbo maximization with evidence maximization. maximizing elbo maximizes a lower bound on $\log p(x)$, not $\log p(x)$ itself. the gap is the kl from $q$ to the true posterior..
- Pitfall: Don't answer "Confusing ELBO maximization with evidence maximization. Maximizing ELBO maximizes a lower bound on $\log P(X)$, not $\log P(X)$ itself. The gap is the KL from $q$ to the true posterior." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Confusing ELBO maximization with evidence maximization. Maximizing ELBO maximizes a lower bound on $\log P(X)$, not $\log P(X)$ itself. The gap is the KL from $q$ to the true post…

### R-hat < 1.01 is necessary but not sufficient for convergence
- Direct Answer: also check ESS and trace plots.
- Why: This matters because it tells you how to reason about r-hat < 1.01 is necessary but not sufficient for convergence.
- Pitfall: Don't answer "R-hat < 1.01 is necessary but not sufficient for convergence" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: also check ESS and trace plots.

### Discarding too few warmup samples. The chain must reach stationarity before samples are useful.
- Direct Answer: Discarding too few warmup samples. The chain must reach stationarity before samples are useful.
- Why: This matters because it tells you how to reason about discarding too few warmup samples. the chain must reach stationarity before samples are useful..
- Pitfall: Don't answer "Discarding too few warmup samples. The chain must reach stationarity before samples are useful." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Discarding too few warmup samples. The chain must reach stationarity before samples are useful.

### Using MCMC without gradient when HMC is available. For continuous targets, HMC dramatically outperforms random-walk methods.
- Direct Answer: Using MCMC without gradient when HMC is available. For continuous targets, HMC dramatically outperforms random-walk methods.
- Why: This matters because it tells you how to reason about using mcmc without gradient when hmc is available. for continuous targets, hmc dramatically outperforms random-walk methods..
- Pitfall: Don't answer "Using MCMC without gradient when HMC is available. For continuous targets, HMC dramatically outperforms random-walk methods." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Using MCMC without gradient when HMC is available. For continuous targets, HMC dramatically outperforms random-walk methods.

### Topic indices are not identifiable
- Direct Answer: topics may be permuted across different runs. Never compare topic $k=3$ from two independent runs.
- Why: This matters because it tells you how to reason about topic indices are not identifiable.
- Pitfall: Don't answer "Topic indices are not identifiable" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: topics may be permuted across different runs. Never compare topic $k=3$ from two independent runs.

### sklearn's LDA uses online variational EM, not collapsed Gibbs
- Direct Answer: it is fast but produces biased estimates. For research-quality topic models, use collapsed Gibbs (gensim).
- Why: This matters because it tells you how to reason about sklearn's lda uses online variational em, not collapsed gibbs.
- Pitfall: Don't answer "sklearn's LDA uses online variational EM, not collapsed Gibbs" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: it is fast but produces biased estimates. For research-quality topic models, use collapsed Gibbs (gensim).

### Forgetting that LDA assumes a bag-of-words representation
- Direct Answer: word order is discarded.
- Why: This matters because it tells you how to reason about forgetting that lda assumes a bag-of-words representation.
- Pitfall: Don't answer "Forgetting that LDA assumes a bag-of-words representation" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: word order is discarded.

### Treating CRF and HMM as equivalent with different parameter estimation. The key distinction is the feature expressiveness
- Direct Answer: CRFs can use overlapping, non-independent features; HMMs fundamentally cannot.
- Why: This matters because it tells you how to reason about treating crf and hmm as equivalent with different parameter estimation. the key distinction is the feature expressiveness.
- Pitfall: Don't answer "Treating CRF and HMM as equivalent with different parameter estimation. The key distinction is the feature expressiveness" by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: CRFs can use overlapping, non-independent features; HMMs fundamentally cannot.

### Forgetting that $Z(X)$ must be recomputed for each sequence in a CRF. This is more expensive than a global $Z$ but is what enables the discriminative conditioning.
- Direct Answer: Forgetting that $Z(X)$ must be recomputed for each sequence in a CRF. This is more expensive than a global $Z$ but is what enables the discriminative conditioning.
- Why: This matters because it tells you how to reason about forgetting that $z(x)$ must be recomputed for each sequence in a crf. this is more expensive than a global $z$ but is what enables the discriminative conditioning..
- Pitfall: Don't answer "Forgetting that $Z(X)$ must be recomputed for each sequence in a CRF. This is more expensive than a global $Z$ but is what enables the discriminative conditioning." by naming the concept alone; state the mechanism and tradeoff.
- Interview line: Say: Forgetting that $Z(X)$ must be recomputed for each sequence in a CRF. This is more expensive than a global $Z$ but is what enables the discriminative conditioning.
