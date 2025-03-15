# db Genie

Overview

* What is DB Genie



What is DB genie ?

* DB defined genie as a compound AI system which is a system that takles AI tasks using multiple interacting components including multiple calls to models, retrievers or external tools.
* This is in contrast to a AI/ LLm model which is simple a statistical model i.e. a transformer that predicts the next token in the text.
* Although the models are improving more and more SOTA results are obtained using compound systems. There are several reasons t=for the same&#x20;
  * some tasks are easier to improve via LLM design
    * suppose that the current best LLM can solve coding contest problems 30% of the time, and tripling its training budget would increase this to 35%; this is still not reliable enough to win a coding contest! In contrast, engineering a system that samples from the model multiple times, tests each sample, etc. might increase performance to 80% with today’s models, as shown in work like [AlphaCode](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf).
  * Systems can be dynamic
    * As LLMs are trained on static dataset, thier knowledge is fixed. Thus developers combine models with other components such as search and retrieval to incorporate timely data.
  * Improving control and trust&#x20;
    * Using an Ai system instead of model helps developers control the model behaviour more tightly by filtering model outputs.
    * LLMs with retrieval can increase user trust by providing citation or automatically verifying facts.
  * Performance goals vary widely -
    * each Ai model has a fixed quality vel and cost but applications need to vary these parameters.
* There are clear benifits with Compound AI systems, but the art of designing, optimizing and operating them is still emerging.&#x20;
* There are many permutations and combinations to try from like should the overall control logic be written in traditional code (python code that&#x20;
