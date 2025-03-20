# RAG

## Overview

* What is RAG?
  *
* Types of RAG
  * Traditional RAG
  * Graph RAG
* Traditional RAG VS Graph RAG
* Summary of RAG eco-system

\-----------------------------------------------------------------------------------------

### What is RAG?

<figure><img src="../.gitbook/assets/image (4).png" alt=""><figcaption></figcaption></figure>

* LLM + external resources which were not included in their training
  * i.e. Any LLM (gpt-4o, Gemini, llama) + internal document, service now ticket etc
* RAG is GenAI design technique that enhances LLM with external knowledge, thus improving the LLMS with&#x20;
  * Proprietary knowledge - It includes proprietary info which wasn't initially used to train the LLMs such as emails, documentation etc. &#x20;
  * Up to date information - RAG application supply LLMs with info from updated data resources.
  * Citing resources - RAG enables LLMs to cite specific resources thus allowing users to verify the factual accuracy of responses.
* RAG includes --
  * Indexing --
    * Data preparation step where data on which retrieval(next step) is performed is extracted and cleaned from data sources and converted into plain text.
    * Ex - If you want to create a RAG for Rx trouble shooting using the email conversation between the end user and CODE orange. You can't pass the entire email conversation to LLM since it might exceeds the context window of the LLM.\
      Hence we break the entire content into smaller and managable pieces called chunks , this process is called chunking.\
      These are then transformed into high dimensional vectors with help of embedding models which gives us a list of chiunk pairs of the data source.
  * Retrieval&#x20;
    * Users request is used to query an outside data store such as Vector store, SQL DB etc. ( here we can use differnet type of DB which leads to different types of DB like tradional rag,Graph based RAG etc.
    * The goal is to get supporting data for LLM's response.
  * Augmentation&#x20;
    * The retrieved data is combined with user's request using a template with additional formatting and instructions to create a prompt.
    * This augmentation can be of three types. Iterative, recursice and Adaptive
    *

        <figure><img src="../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>
  * Generation&#x20;
    * The prompt is passed to the LLMs which then generates a response to the query
* Types of RAG
  * Traditional RAG
  * Graph RAG
* Traditional RAG VS Graph RAG



Summary of RAG ecosystem

<figure><img src="../.gitbook/assets/image (3) (1).png" alt=""><figcaption></figcaption></figure>

