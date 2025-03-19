# How to train Your Dragon (LLM)

<figure><img src="../.gitbook/assets/image (5).png" alt=""><figcaption></figcaption></figure>

Overview:

* What is a LLM?
* Transform artitecture
* Overall process

## What is a LLM?

* Large language models
* New era for NLP as earlier we had traditional methods which underperformed in tasks that demanded complex understanding and generation abilities.
* ex— they were not able to write email from a list of keywords.
* LLMs are trained on vast quantites of data, the sucess lies behind transformer architecture that underpins many LLMs and vast amount of data on which LLMs are trained on which allows them to capture wide variety of linguistic nuances, contexts and patterns that is challenging to encode manually.
* These have billions of parameters which are adjustable weights in the network that are optimized during training to predict the next word in the sequece.
* The transformer artitecture allows them to pay selective attention to different parts of input when making predictions making them especially adept at handling the nuances and complexities of human language.
* LLMs can be categorised as an intersection between Deep learning and GenAI.
* custom built LLMs those are tailored for specific tasks or domains can outperform general purpose LLMs such as ChatGPT.
* THe pretrained model ChatGPT serve as a foundational resource that can be further refined through fine tuning, a process where the model is specifically trained on a narrower dataset that is more specific to a particular tasks.
* LLMs user self-supervised learning where the model generated its own labels from the input data.

## Transformer artitecture

* Transformer consists of two submodules: an encoder and a decoder.
  * encoder&#x20;
    * processes the input text and encodes it into a series of numerical representations or vectors that capture the contextual information of the input.
  * Decoder&#x20;
    * takes the encoded vectors and generated the output text.
* Both the encoder and decoder consists of many layers connected by so called self-attention mechanism, which allows the model to weigh the importatnce of difference words or toekns in a sequence relative to each other.
* this enables the model to cpture long rage dependencies and contextual relationships withing the input data, enhancing its ability to generate coherent and contextually relevant output.
* Steps
  * encoder&#x20;
    * input text — input text to be translated
    * preprocessing steps — input text is prepared for the encoder.
    * encoder then produces the text encodings used by the decoder
    * encoder returns the embedding vectors as input to the decoder.
  * Decoder
    * A partial output test, the model completes the translation one word at a time
    * The input text is prepared for the decoder
    * The decoder generates the translated text one word at a time
    * the complete output
* There are many LLMs each having different training approach from GPT. GPT is designed for generative tasks, BERT specializes in masked word prediction  where the model predicts the masked or hidden word in a given sequence.
* &#x20;

## Overall Process

1. Building Your LLM
   1. Data preparation and sampling
   2. Attention Mechanism
      1. Simplified self-attention
      2. self-attention
      3. Casual attention
      4. Multi-head attention
   3. LLM architecture
      1.
2. Foundation model (pretraining)
   1. Training loop
   2. Model evaluation
   3. Load pretrained weights
3. Fine tunining
