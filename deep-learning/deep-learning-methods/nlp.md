# NLP

Here are detailed notes on the models used in Natural Language Processing (NLP), from foundational statistical methods to modern Transformer architectures.

#### 📜 What is an NLP Model?

In NLP, a "model" is a system designed to understand, interpret, and generate human language. The primary challenge is converting "words" (which are ambiguous and context-dependent) into a numerical format that a computer can process.

The evolution of these models is a story of capturing context:

* Level 1 (Statistical): Do these words _appear_ in the document?
* Level 2 (Word Embeddings): What words _neighbor_ this word?
* Level 3 (Sequential): What is the _order_ of the words?
* Level 4 (Transformers): What is the _relationship_ of _every word_ to _every other word_?

***

#### 1. 📊 Foundational Statistical Models

These models are "classic," fast, and work well for simple tasks. They treat text as a "bag" or collection of words, ignoring order.

**### Bag-of-Words (BoW)**

* How it Works: Creates a vector for a document by counting the frequency of every word.
* Use Case: Basic text classification (e.g., spam vs. not-spam), document clustering.
* Pros:
  * Very simple, fast, and easy to understand.
  * Works surprisingly well as a baseline for simple problems.
* Cons:
  * No context: Ignores word order and grammar ("the dog bit the man" is identical to "the man bit the dog").
  * Huge, sparse vectors: The vector size is the entire vocabulary (100,000+ words), and most entries are zero.
  * Gives too much weight to common words (like "the," "is," "a").

**### TF-IDF (Term Frequency - Inverse Document Frequency)**

* How it Works: An upgrade to BoW. It still counts words (Term Frequency) but then _down-weights_ words that are common across _all_ documents (Inverse Document Frequency).
* Core Idea: A word is important if it appears _frequently_ in one document but _rarely_ in all other documents.
* Use Case: The classic algorithm for search engines, document ranking, and keyword extraction.
* Pros:
  * Much better than BoW at finding _relevant_ and _topical_ words.
  * Still fast and simple to compute.
* Cons:
  * Still no context: Word order and semantics are ignored. "Good" and "excellent" are treated as completely different things.

***

#### 2. 💡 Word Embedding Models

These models were a breakthrough, as they learned to represent the _meaning_ and _relationships_ of words.

**### Word2Vec & GloVe**

* How it Works: Both are algorithms that create a "word embedding"—a dense vector (e.g., 300 dimensions) for every word in the vocabulary. Words with similar meanings are "plotted" close together in this vector space.
* Key Feature: They enable vector math for words. The classic example is: `Vector("King") - Vector("Man") + Vector("Woman") ≈ Vector("Queen")`.
* Use Case: The standard input for _all_ modern neural NLP models (RNNs, LSTMs, Transformers). They "embed" words before the network processes them.
* Pros:
  * Captures semantics: "Cat" and "Dog" are close, while "Cat" and "Car" are far.
  * Efficient: The vectors are dense (not sparse) and relatively small.
* Cons:
  * No polysemy: The word "bank" (river bank vs. financial bank) has only _one_ vector, so its meaning is "averaged out" and ambiguous.
  * Static: The embeddings are pre-trained and don't change based on the sentence's context.

***

#### 3. ➡️ Sequential Models

These models were the first to treat language as a _sequence_, where order matters.

**### RNN, LSTM, & GRU**

* RNN (Recurrent Neural Network): A layer with a "loop" that processes one word at a time, passing a "memory" (hidden state) to the next step.
* LSTM & GRU (Long Short-Term Memory & Gated Recurrent Unit): Sophisticated RNNs that use "gates" to control this memory. They can decide what to _forget_, what to _remember_, and what to _output_.
* Use Case: Were the state-of-the-art for translation, sentiment analysis, and time-series data.
* Pros:
  * Handles word order: "Man bit dog" is now different from "dog bit man."
  * LSTMs/GRUs can capture long-range dependencies (e.g., connecting a subject at the start of a sentence to a verb at the end).
* Cons:
  * Vanishing Gradients: Simple RNNs "forget" information from just a few steps back.
  * Slow & Sequential: Cannot be parallelized. To process the 10th word, you _must_ have processed the first 9. This makes training on massive datasets very difficult.

Here is a detailed explanation of the inner workings of RNNs, LSTMs, and GRUs.

***

#### 1. Simple Recurrent Neural Network (RNN)

This is the most basic version and the foundation for the others.

**💡 The Core Idea: The Loop**

An RNN cell processes one item from a sequence (e.g., one word) and combines it with a "memory" of the _previous_ item. It then passes this updated memory to the next step.

Think of it as a person reading a sentence one word at a time, constantly updating their "summary" of what they've read so far.

**⚙️ How It Works**

At each time step `t`, the RNN cell performs two simple tasks:

1. Calculate the Hidden State (The "Memory"):
   * It takes the current input $$ $x_t$ $$ (e.g., the vector for the word "cat").
   * It also takes the _hidden state from the previous step_, $$ $h_{t-1}$ $$ (the "memory" of the sentence so far).
   * It combines them, puts them through an activation function (like `tanh`), and produces the new hidden state, $$ $h_t$ $$.
   * $$ $h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$ $$
2. Make a Prediction (Optional):
   * It can use this new hidden state $$ $h_t$ $$ to make an output $$ $y_t$ $$ (e.g., predict the _next_ word).
   * $$ $y_t = W_{hy} \cdot h_t + b_y$ $$

This new hidden state $$ $h_t$ $$ is then passed as the "memory" to the next time step, $$ $t+1$ $$. The _same set of weights_ ($$ $W_{xh}$ $$, $$ $W_{hh}$ $$, etc.) is used at every single step.

**❌ The Fatal Flaw: The Vanishing Gradient Problem**

An RNN's memory is very short. To train the network, you use "backpropagation through time," which is just regular backpropagation unrolled across the sequence.

* To update the weights, gradients are multiplied at every single time step.
* If the gradient is a small number (e.g., 0.9), after 50 time steps, the final gradient is $$ $0.9^{50} \approx 0.005$ $$. It's _vanished_.
* This means the network is _incapable_ of learning from long-range dependencies. It can't learn to connect the word "barked" at the end of a long paragraph back to the word "dog" at the beginning.

***

#### 2. Long Short-Term Memory (LSTM)

LSTMs were designed _specifically_ to solve the vanishing gradient problem.

**💡 The Core Idea: A "Conveyor Belt" Memory**

An LSTM introduces a dedicated, separate "memory line" called the Cell State ($$ $c_t$ $$). Think of this as a conveyor belt that carries information down the sequence.

The LSTM has special "gates" that can _learn_ to add information to this belt, or remove information from it. This system gives it a stable, long-term memory.

**⚙️ How It Works**

An LSTM cell has two states it passes to the next step:

* Cell State ($$ $c_t$ $$): The long-term memory ("conveyor belt").
* Hidden State ($$ $h_t$ $$): The short-term memory / working state (used for the current output).

It uses three "gates" (which are just small sigmoid neural networks) to control this memory. A sigmoid function outputs a number between 0 (block everything) and 1 (let everything through).

1. 🚪 Forget Gate:
   * Question: "What parts of the _old_ long-term memory ($$ $c_{t-1}$ $$) should I forget?"
   * How: It looks at the new input $$ $x_t$ $$ and the last hidden state $$ $h_{t-1}$ $$.
   * Example: If it sees a new sentence subject (e.g., "A new dog..."), it might learn to "forget" the _previous_ sentence's subject. It outputs a "forget vector" (e.g., `[1, 1, 0, ...]`) to multiply with $$ $c_{t-1}$ $$.
2. 🚪 Input Gate:
   * Question: "What _new_ information from the current input should I add to the long-term memory?"
   * How: It has two parts:
     * An "input" sigmoid gate decides _which_ values to update.
     * A `tanh` layer creates a "candidate" vector of new information ($$ $\tilde{c}_t$ $$) to be added.
   * Example: If it sees the word "dog," the candidate vector is the "dog" information. The input gate decides to "add" this information.
3. 🚪 Output Gate:
   * Question: "What part of my long-term memory is relevant for my output _right now_?"
   * How: It looks at the new input $$ $x_t$ $$ and last hidden state $$ $h_{t-1}$ $$ to decide what to output from the _newly updated_ cell state $$ $c_t$ $$.
   * Example: The cell state might hold "brown dog, female." The _current_ task might only need to know "dog." The output gate learns to filter the cell state and produce the final hidden state $$ $h_t$ $$ (the "working memory").

Why this works: The cell state "conveyor belt" has very simple math (just addition and multiplication). This allows the gradient to flow back _almost unchanged_, protected by the gates. The gates _learn_ when to open and close, so the gradient doesn't vanish.

***

#### 3. Gated Recurrent Unit (GRU)

A GRU is a (newer) simplified version of an LSTM. It's the "sleek, modern" version that achieves the same goal with less complexity.

**💡 The Core Idea: Combine and Simplify**

A GRU works by merging the LSTM's Cell State and Hidden State into a _single_ state $$ $h_t$ $$. It also combines the "forget" and "input" gates into a single gate.

**⚙️ How It Works**

A GRU has only two gates:

1. 🚪 Reset Gate ($$ $r_t$ $$):
   * Question: "How much of the _past_ memory should I ignore when creating my new 'candidate' memory?"
   * How: It looks at the new input $$ $x_t$ $$ and last hidden state $$ $h_{t-1}$ $$.
   * Action: This gate decides how much of $$ $h_{t-1}$ $$ to use. If $$ $r_t$ $$ is 0, it _completely ignores_ the past memory, effectively "resetting" for a new context.
2. 🚪 Update Gate ($$ $z_t$ $$):
   * Question: "What's the balance? How much of the _old_ memory ($$ $h_{t-1}$ $$) should I keep, and how much of the _new_ candidate memory ($$ $\tilde{h}_t$ $$) should I add?"
   * How: This is the key. It outputs a single value $$ $z_t$ $$ (e.g., 0.8).
   * Action: It uses $$ $z_t$ $$ to control the new memory:
     * $$ $h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t$ $$
   * Example: If $$ $z_t = 0.8$ $$, the new state is 80% new information and 20% old information. If $$ $z_t = 0.1$ $$, the new state is 10% new information and 90% old information (it's just "carrying over" the old memory).

***

#### 🚀 Summary: LSTM vs. GRU vs. RNN

| **Feature**  | **Simple RNN**                                        | **LSTM (Long Short-Term Memory)**                                                      | **GRU (Gated Recurrent Unit)**                                                  |
| ------------ | ----------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Core Idea    | A simple loop/memory.                                 | Gated "conveyor belt" for memory.                                                      | A simplified, faster version of LSTM.                                           |
| Memory       | One "hidden state" $$ $h_t$ $$.                       | Two states: Cell State $$ $c_t$ $$ (long-term) and Hidden State $$ $h_t$ $$ (working). | One "hidden state" $$ $h_t$ $$.                                                 |
| No. of Gates | 0                                                     | 3 (Forget, Input, Output)                                                              | 2 (Reset, Update)                                                               |
| Key Problem  | Vanishing Gradient. Cannot learn long-range patterns. | Solves the vanishing gradient problem.                                                 | Solves the vanishing gradient problem.                                          |
| Speed        | Fast (but useless).                                   | Slower (most complex).                                                                 | Faster (fewer parameters than LSTM).                                            |
| When to Use  | Never. (Only for teaching).                           | The default standard. Good for complex tasks where you need max performance.           | A great first choice. Often performs _just as well_ as LSTM, but trains faster. |

***

#### 4. 🚀 Transformer-Based Models (The Modern Era)

This architecture (from the 2017 paper "Attention Is All You Need") revolutionized NLP by getting rid of sequential processing and using self-attention.

Core Idea: Instead of a slow, one-word-at-a-time loop, self-attention allows the model to look at _all_ words in the sentence at once and calculate a "relevance score" for every word relative to every other word.

**### BERT (Encoder-Only)**

* Stands For: Bidirectional Encoder Representations from Transformers.
* How it Works: It's a "stack of encoders" designed to _understand_ text. It reads the entire sentence at once, using self-attention to see both left-and-right context (it's "bidirectional").
* Training: It's pre-trained by "masking" (hiding) 15% of the words in a sentence and learning to predict them.
* Use Case (NLU - Natural Language Understanding):
  * Sentiment Analysis & Text Classification: Is this review positive or negative?
  * Question Answering (QA): Given a paragraph, find the _span_ of text that answers a question.
  * Search Engines: The engine behind Google Search.
* Pros:
  * Deep Contextual Understanding: It's the "king" of understanding. It solves the "bank" problem (it knows if it's a river bank or a money bank).
  * State-of-the-art on all understanding-based benchmarks (like the GLUE benchmark).
* Cons:
  * Not a text generator: It's an "encoder," not a "decoder." It's not designed to write new text.
  * Very large and computationally expensive.

**### GPT (Decoder-Only)**

* Stands For: Generative Pre-trained Transformer.
* How it Works: It's a "stack of decoders" designed to _generate_ text. It's "autoregressive," meaning it only looks at the text from _left-to-right_ (it can't "see the future").
* Training: It's pre-trained on a simple task: predict the next word.
* Use Case (NLG - Natural Language Generation):
  * Chatbots & Conversational AI: (e.g., ChatGPT)
  * Content Creation: Writing articles, code, marketing copy.
  * Text Completion
* Pros:
  * Incredibly fluent and human-like text generation.
  * "In-context learning": You can "program" it with a prompt (e.g., "Translate this text...") without re-training it.
* Cons:
  * Can "hallucinate": It's trained to be _plausible_, not _truthful_. It will confidently make up facts.
  * Its "one-way" (unidirectional) nature makes it less suitable for deep understanding tasks than BERT.

**### T5 (Encoder-Decoder)**

* Stands For: Text-to-Text Transfer Transformer.
* How it Works: A "complete" Transformer that has both an Encoder (to understand input) and a Decoder (to generate output).
* Core Idea: It reframes _every_ NLP task as a text-to-text problem.
  * Translation: `"translate English to German: The cat is."` $$ $\rightarrow$ $$ `"Die Katze ist."`
  * Summarization: `"summarize: [long article]..."` $$ $\rightarrow$ $$ `"[short summary]..."`
  * Classification: `"classify: This movie was great."` $$ $\rightarrow$ $$ `"positive"`
* Use Case: A "Swiss Army Knife" for any task that transforms one sequence into another.
* Pros:
  * Extremely versatile: A single model can be fine-tuned to do _anything_.
  * Excellent for summarization, translation, and Q\&A.
* Cons:
  * Often "overkill" for a simple classification task where BERT would be more efficient.
  * Can be larger and more complex to train than an encoder-only or decoder-only model.
