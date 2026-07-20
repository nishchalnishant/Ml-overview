---
module: Data
topic: Overview
subtopic: ""
status: unread
tags: [data, eda, feature-engineering, index]
prerequisites: [statistics]
---
# Data

Everything between raw data and a model-ready matrix. In interviews this section is underrated: most candidates rush to model choice, while practitioners know the data pipeline is where accuracy is actually won or lost.

**What lives here:** EDA, preprocessing, feature engineering, leakage, validation splits, imbalance.

**The highest-frequency trap in this section** is data leakage — fitting a scaler or an encoder on the full dataset before splitting. It inflates validation scores and is the single most common reason a model that looked strong offline collapses in production.
