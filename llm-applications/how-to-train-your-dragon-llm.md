# How LLMs Actually Work

This file is the under-the-hood guide for people who want the real mechanics without drowning in research-paper fog.

Think of it as the Azure engineer's version of LLM internals:

- what gets built
- what runs at inference time
- where the cost goes
- why the model feels smart one minute and strange the next

## The One-Line Answer

A modern LLM is usually a **decoder-only Transformer** trained to predict the next token from the tokens that came before it.

That is the whole engine.

Everything impressive comes from doing that at huge scale with strong architecture, lots of data, and expensive compute.

## Why Next-Token Prediction Becomes Powerful

It sounds almost suspiciously simple.

But if a model gets very good at predicting what comes next across massive text, code, and structured patterns, it starts learning:

- grammar
- facts
- style
- reasoning patterns
- code structure
- conversational rhythm

It is a bit like listening to thousands of old romantic songs and then being able to predict the next line because the meter, mood, and emotional arc are deeply internalized.

Not magic. Just a lot of pattern learning.

## Attention: The Star of the Show

Attention is how one token decides which previous tokens matter most.

The classic trio:

- **Query** = what this token is looking for
- **Key** = what earlier tokens can offer
- **Value** = the information they contribute

The model compares query-key similarity, then uses the matching values more heavily.

That is how the model can tell whether "bank" means finance or river edge.

## Ghazal Analogy

In a Gulzar ghazal, a single word can sound playful, aching, or devastating depending on the lines that came before it.

That is attention in spirit.

Same word.
Different context.
Different meaning.

## Why Scale by `sqrt(d_k)`

As vector dimensions grow, raw attention dot products can become too large.

If that happens:

- softmax becomes too sharp
- tiny score differences dominate too hard
- gradients become less stable

So we divide by `sqrt(d_k)` to keep the numbers in a healthier range.

It is basically numerical discipline, not decorative math.

## Positional Information

Transformers process tokens in parallel.

Great for speed.
Terrible for word order if left alone.

So they need positional information to understand sequence.

Without that, these would look far too similar:

- "Mumbai chased brilliantly after the rain break"
- "After the rain break Mumbai brilliantly chased"

Same words, different flow.

Language cares about order. So does cricket commentary.

## Tokens: What the Model Really Reads

Models do not read words the way humans do.

They read **tokens**, which may be:

- full words
- subwords
- punctuation fragments
- byte-level pieces

Why this matters in real systems:

- token count affects cost
- token count affects latency
- token count affects context-window usage

This is why prompt design is also resource design.

## BPE and WordPiece

Word-level vocabularies are too big.
Character-level sequences are too long.

So subword tokenization is the practical middle path.

That lets the model handle rare words, names, and mixed text efficiently without exploding either vocabulary size or sequence length.

## Scaling Laws

A larger model often performs better, but only if the rest of the setup scales sensibly too.

You usually need balance across:

- parameters
- data
- compute

This is the big lesson behind scaling-law thinking: bigger helps, but wasteful imbalance does not.

If you train a huge model on weak or insufficient data, you are not being ambitious. You are just lighting budget on fire with confidence.

## Azure / DevOps Bridge

Think of large-scale training like a very expensive pipeline stage:

- huge infra bill
- long runtime
- artifact produced at the end
- lots of pressure to version everything properly

Inference is the opposite side of the house:

- low latency pressure
- autoscaling concerns
- throughput optimization
- cache strategy
- runtime safety

Training is build time.
Inference is production runtime.

That distinction matters a lot in interviews.

## Inference: How the Model Generates

At inference, the model generates one token at a time.

The loop is basically:

1. take current context
2. predict probability distribution over next token
3. choose one token
4. append it
5. repeat

This sequential nature is why generation latency matters so much. Even if the model is massively parallel inside each step, the output still comes out token by token.

## Decoding Strategies

Once the model gives logits, you still have to choose how to turn them into tokens.

Common options:

- **greedy**: always pick the top token
- **beam search**: keep multiple promising paths
- **top-k**: sample from top candidates
- **top-p**: sample from the smallest set covering enough probability mass
- **temperature**: control randomness

Low temperature is like singing exactly to notation.

Higher temperature is like improvising in a live mehfil.

Sometimes enchanting.
Sometimes slightly off-key.

## KV Cache

KV cache is one of the most practical LLM inference ideas.

Without it, the model keeps recomputing attention state for earlier tokens at every generation step.

With it:

- past keys are stored
- past values are stored
- only the new token needs fresh work

That is a major latency win.

## Azure Parallel

KV cache feels a lot like reusing pipeline outputs or cached layers instead of rebuilding everything from scratch on every run.

Same principle:

- do expensive work once
- reuse it aggressively

## Perplexity

Perplexity measures how surprised the model is by real text.

Lower perplexity usually means better next-token prediction under the same setup.

Useful?
Yes.

Enough for product quality?
No.

You still need to care about:

- instruction following
- groundedness
- safety
- factual reliability
- task success

## Encoder-Only vs Decoder-Only vs Encoder-Decoder

Keep this answer clean in interviews.

- **encoder-only** models are strong at understanding tasks like classification
- **decoder-only** models are strong at generation and chat
- **encoder-decoder** models are strong at input-to-output transformations like translation or summarization

Simple. Clear. Strong.

## What Actually Makes an LLM App Feel Smart

Not just bigger weights.

Usually it is a combination of:

- strong base model
- careful prompting
- retrieval
- good tool use
- validation
- sensible UX

The model is the singer.
The rest of the stack is the orchestra, sound engineer, hall acoustics, and mastering.

## Mini Pop Quiz

Why does KV cache matter more in production than in a whiteboard explanation?

Because real systems care about latency and cost, not just architectural elegance.

## How Would You Explain This in an Interview?

Try this:

"An LLM is usually a decoder-only Transformer trained for next-token prediction. Attention lets each token weigh previous context, tokenization defines how text becomes model-readable units, and inference generates one token at a time. In production, practical topics like KV caching, decoding strategy, latency, and context-window management matter as much as the core architecture."

That answer sounds grounded, modern, and engineering-aware.
