# Deep Learning for Time Series

Time-series modeling is where deep learning has to respect something ordinary tabular workflows often ignore:

time is not just another column.

Order matters.
Leakage gets sneaky.
Validation gets stricter.

---

# 1. Classical vs Deep Approaches

Classical models like:

- ARIMA
- Prophet

are still very useful when:

- patterns are simpler
- interpretability matters
- data volume is limited

Deep learning becomes attractive when:

- sequences are complex
- multiple signals interact
- non-linear temporal structure matters

---

# 2. RNNs and LSTMs for Time Series

RNN-style models were natural early choices because they handle order directly.

They can model:

- trends
- memory
- sequential dependence

But they also inherit the usual recurrent issues:

- slower training
- vanishing gradients

---

# 3. Transformers for Time Series

Transformers can work well for time series when long-range dependencies matter and enough data exists.

But they are not automatic upgrades.

They come with:

- more compute cost
- more data appetite
- more tuning sensitivity

So the strongest answer is usually balanced:

use the simplest method that fits the temporal complexity and deployment constraints.
