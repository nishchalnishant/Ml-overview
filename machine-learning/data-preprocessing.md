# Data preprocessing

Data preprocessing is the set of steps taken to clean and transform raw data into a clean, well-formatted dataset that can be fed into a machine learning model.

It's arguably the most important stage of any machine learning project. The quality of your data directly determines the quality of your model, following the "Garbage In, Garbage Out" (GIGO) principle.

***

#### A Critical Step _Before_ You Begin: The Train-Test Split

Before you do _any_ preprocessing, you **must split your data into training and testing sets.**

* Why? This prevents data leakage.
* Data Leakage is when information from your test set (which is supposed to be "unseen" data) "leaks" into your training process.
* Example: If you calculate the mean and standard deviation of your _entire_ dataset and then use it to scale all your data (a process called Standardization), your training set has now been influenced by the values in your test set.
* Correct Way:
  1. Split your data into `X_train`, `X_test`, `y_train`, `y_test`.
  2. Fit your preprocessor (e.g., `StandardScaler`, `CountVectorizer`) only on the `X_train` data.
  3. Use that _same_ fitted preprocessor to transform both `X_train` and `X_test`.

***

## <mark style="color:blue;">1. Preprocessing for Tabular Data (Numerical & Categorical)</mark>

This is the most common data type, found in spreadsheets or databases. It's a mix of number and text columns.

### <mark style="color:green;">**General Cleaning (Apply to all)**</mark>

* <mark style="color:yellow;">**Handling Missing Data:**</mark>
  * <mark style="color:red;">Deletion</mark>: Remove any rows (listwise deletion) or columns that have missing values.
    * Pro: Simple.
    * Con: You lose data. Bad if you have few samples or if the data is "missing not at random."
  * <mark style="color:red;">Imputation</mark>: Fill in the missing values.
    * Numerical: Fill with the mean, median (best for skewed data), or a constant (like 0).
    * Categorical: Fill with the mode (most frequent value) or a constant (like "Missing").
* <mark style="color:yellow;">**Handling Duplicates:**</mark>
  * <mark style="color:red;">Remove duplicate rows (</mark><mark style="color:red;">`drop_duplicates()`</mark><mark style="color:red;">).</mark> They add no new information and can bias the model.
* <mark style="color:yellow;">**Handling Outliers:**</mark>
  * <mark style="color:red;">Remove</mark>: If it's a clear data entry error.
  * <mark style="color:red;">Cap (Clipping):</mark> Set all values above a certain percentile (e.g., 99th) to that value.
  * <mark style="color:red;">Transform</mark>: Use a log transform to pull in extreme values.

### <mark style="color:green;">**For Numerical Features**</mark>

Models are often sensitive to the _scale_ of those numbers. A "Salary" (10,000-100,000) will overpower an "Age" (20-60).

* <mark style="color:yellow;">**Feature Scaling (Standardization):**</mark>
  * What: Rescales the data to have a <mark style="color:red;">mean of 0 and a standard deviation of 1.</mark>
  * Formula: $$ $z = (x - \mu) / \sigma$ $$
  * When: The default, go-to scaling method. Required for models like SVM, Logistic Regression, and PCA.
* <mark style="color:yellow;">**Feature Scaling (Normalization):**</mark>
  * What: <mark style="color:red;">Rescales the data to a specific range, usually</mark> <mark style="color:red;"></mark><mark style="color:red;">`[0, 1]`</mark><mark style="color:red;">.</mark>
  * Formula: $$ $x_{\text{norm}} = (x - x_{\text{min}}) / (x_{\text{max}} - x_{\text{min}})$ $$
  * When: Good for image data (pixel intensities) and neural networks.
* <mark style="color:yellow;">**Log Transform:**</mark>
  * What<mark style="color:red;">: Applies</mark> $$ $log(x)$ $$ <mark style="color:red;">to the data.</mark>
  * When: Used to handle highly skewed data (e.g., incomes, website traffic). It makes the distribution more "normal" (Gaussian).

### <mark style="color:green;">**For Categorical Features**</mark>

Models don't understand text like "Red," "Green," or "Blue." You must encode them into numbers.

* <mark style="color:yellow;">**Label Encoding:**</mark>
  * What: <mark style="color:red;">Converts each unique category into an integer</mark>. (e.g., "Red" $$ $\rightarrow$ $$ 0, "Green" $$ $\rightarrow$ $$ 1, "Blue" $$ $\rightarrow$ $$ 2).
  * When: Only for ordinal data, where the order matters (e.g., "Small" $$ $\rightarrow$ $$ 0, "Medium" $$ $\rightarrow$ $$ 1, "Large" $$ $\rightarrow$ $$ 2).
  * Con: Creates a false mathematical relationship (Blue > Green) that will confuse most models.
* <mark style="color:yellow;">**One-Hot Encoding (OHE):**</mark>
  * What: <mark style="color:red;">Creates a new binary (0/1) column for</mark> <mark style="color:red;"></mark>_<mark style="color:red;">each</mark>_ <mark style="color:red;"></mark><mark style="color:red;">unique category</mark>.
    * "Red" $$ $\rightarrow$ $$ `[1, 0, 0]`
    * "Green" $$ $\rightarrow$ $$ `[0, 1, 0]`
    * "Blue" $$ $\rightarrow$ $$ `[0, 0, 1]`
  * When: The default for nominal data (where order _doesn't_ matter).
  * Con: Can create a _lot_ of new columns (the "curse of dimensionality") if you have a category with 10,000 unique values.

***

## <mark style="color:blue;">2. Preprocessing for Text Data (NLP)</mark>

The goal is to turn sentences into numerical vectors (a process called vectorisation).

* <mark style="color:yellow;">**Cleaning**</mark>**:**
  * <mark style="color:red;">Lowercasing</mark>: Converts all text to lowercase ("The" and "the" become the same).
  * <mark style="color:red;">Removing Punctuation</mark>: Removes all `.,!?"` etc.
  * <mark style="color:red;">Removing Stop Words</mark>: Removes common words that add little meaning (e.g., "a", "an", "the", "is", "in").
* <mark style="color:yellow;">**Tokenization**</mark>**:**
  * <mark style="color:red;">What</mark>: Splits a sentence into a list of individual words (tokens).
  * Example: "The cat sat" $$ $\rightarrow$ $$ `['the', 'cat', 'sat']`.
* <mark style="color:yellow;">**Normalization**</mark>:
  * <mark style="color:red;">Stemming</mark>: A crude method of chopping off word endings. (e.g., "running", "runs" $$ $\rightarrow$ $$ "run"). Fast but can be inaccurate.
  * <mark style="color:red;">Lemmatization</mark>: A smarter method that uses a dictionary to find the root form of a word. (e.g., "ran" $$ $\rightarrow$ $$ "run", "better" $$ $\rightarrow$ $$ "good"). Slower but more accurate.
* <mark style="color:yellow;">**Vectorization**</mark>:
  * <mark style="color:orange;">Bag-of-Words (BoW) / CountVectorizer</mark>: Creates a vector where each entry is a _count_ of how many times a word appeared in the text. Ignores grammar and word order.
  * <mark style="color:red;">TF-IDF Vectorizer</mark>: An upgrade to BoW. It counts words but _down-weights_ common words (like "the") and _boosts_ the score of rare, more meaningful words.
  * <mark style="color:red;">Word Embeddings (Word2Vec, GloVe):</mark> The modern approach. Maps each word to a dense vector that captures its _meaning_ and _context_. (e.g., the vectors for "cat" and "dog" will be mathematically "close" to each other).

***

## <mark style="color:blue;">3. Preprocessing for Image Data (CV)</mark>

The goal is to turn a collection of images (which are 3D arrays of pixel values) into a standardized batch of tensors.

* <mark style="color:yellow;">Resizing & Cropping:</mark>
  * <mark style="color:red;">Resize</mark>: _This is mandatory._ All images in a batch must be the exact same height and width (e.g., 224x224) to be fed into the model.
  * <mark style="color:red;">Crop</mark>: (e.g., Center Crop, Random Crop). Used to focus on the important part of an image.
* <mark style="color:yellow;">Normalization / Scaling:</mark>
  * _This is also mandatory._ Pixel values are typically `[0-255]`. This range is too large.
  * Option 1: Normalize to `[0, 1]`: Simply divide all pixels by 255.
  * Option 2: Standardize to `[-1, 1]`: Use the mean and standard deviation of a large dataset (like ImageNet) to standardize the pixels. This is the most common method for transfer learning.
* <mark style="color:yellow;">Data Augmentation (A training-only step):</mark>
  * What: Artificially creating "new" images from your existing ones to make your model more robust and prevent overfitting.
  * Techniques:
    * Geometric: Randomly flipping (horizontally), rotating, zooming, or shearing the image.
    * Color: Randomly changing the brightness, contrast, or saturation.

***

Here are detailed notes on _when_ to use specific preprocessing techniques, along with their pros and cons.

The most important rule is to always fit your preprocessor on the training data only and then use it to transform both your training and test sets. This prevents data leakage.

***

## <mark style="color:blue;">1. Handling Missing Data</mark>

<mark style="color:yellow;">**Deletion (Dropping Rows/Columns)**</mark>

* When to Use:
  * If you have a very large dataset and only a tiny fraction of rows have missing data (e.g., < 1% of rows).
  * If a specific column (feature) is missing a huge amount of data (e.g., > 50-60%) and is not critical.
* Pros:
  * Simple: Very easy and fast to implement.
  * No Bias: Doesn't introduce any "fake" or "artificial" data.
* Cons:
  * Loss of Data: This is the biggest drawback. You are throwing away valuable information.
  * Can Cause Bias: If the data is _not_ missing randomly (e.g., "people with high incomes didn't answer"), deleting rows will bias your entire dataset.

***

<mark style="color:yellow;">**Imputation (Filling Values)**</mark>

* When to Use: The standard approach when you cannot afford to lose data.

| **Method** | **When to Use**                                                                              | **Pros**                                                          | **Cons**                                                                                            |
| ---------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Mean       | For numerical columns that are normally distributed (no significant outliers).               | Very simple and fast.                                             | Extremely sensitive to outliers. A few extreme values will pull the mean and give a bad imputation. |
| Median     | Default choice for numerical columns. Use this when the data is skewed or you have outliers. | Robust to outliers. The median is not affected by extreme values. | Still a simple estimate that doesn't account for relationships between features.                    |
| Mode       | Default choice for categorical columns. (e.g., "Color", "City").                             | Simple, fast, and makes sense for non-numeric data.               | Can bias the model if one category is overwhelmingly dominant.                                      |

***

## <mark style="color:blue;">2. Handling Numerical Data</mark>

<mark style="color:yellow;">**Log Transform**</mark>

* When to Use:
  * When a numerical feature is highly skewed (e.g., has a "long tail" like income, website visits, or housing prices).
* Pros:
  * Pulls in extreme outliers, making the distribution more "normal" (Gaussian).
  * Can help linear models (like Linear Regression) find a better fit.
* Cons:
  * Cannot be used on zero or negative values (you must use $$ $log(x+1)$ $$ instead).
  * Makes the feature less interpretable (you're now in "log-dollars," not "dollars").

***

<mark style="color:yellow;">**Feature Scaling**</mark>

You must scale features for models that are sensitive to distance or magnitude, like Linear Regression, Logistic Regression, SVM, k-NN, and Neural Networks.

<table data-header-hidden><thead><tr><th width="151"></th><th></th><th></th><th></th></tr></thead><tbody><tr><td><strong>Scaler</strong></td><td><strong>When to Use</strong></td><td><strong>Pros</strong></td><td><strong>Cons</strong></td></tr><tr><td><mark style="color:red;">Standardization</mark> (StandardScaler)</td><td>The default, go-to scaler. Use this when your data is normally distributed (or close to it) and you have <em>few</em> outliers.</td><td>Centers the data at mean=0. Required by many classic models.</td><td>Highly sensitive to outliers. The mean and standard deviation are both pulled by extreme values, which can "squish" the rest of your data.</td></tr><tr><td><mark style="color:red;">Normalisation</mark> (MinMaxScaler)</td><td>When you need a strict <code>[0, 1]</code> range. Common for image data (pixel intensities) or neural networks that require inputs in this range.</td><td>Guarantees a fixed range, which is good for some algorithms.</td><td>Extremely sensitive to outliers. A single outlier will become the new <code>min</code> or <code>max</code>, "squishing" all other data into a tiny part of the range.</td></tr><tr><td><mark style="color:red;">Robust Scaling</mark> (RobustScaler)</td><td>The best choice when your data has outliers. Use this when you don't want to remove the outliers but need to scale around them.</td><td>Robust to outliers. It uses the median and Interquartile Range (IQR), which are not affected by extreme values.</td><td>Does not center the data at mean=0. The resulting range is not fixed, which may be a problem for some models.</td></tr></tbody></table>

***

## <mark style="color:blue;">3. Handling Categorical Data</mark>

<mark style="color:yellow;">**Label Encoding**</mark>

* What it is: "Red" $$ $\rightarrow$ $$ 0, "Green" $$ $\rightarrow$ $$ 1, "Blue" $$ $\rightarrow$ $$ 2
* When to Use:
  * Only for Ordinal Data: When there is a clear, meaningful order (e.g., "Small" $$ $\rightarrow$ $$ 0, "Medium" $$ $\rightarrow$ $$ 1, "Large" $$ $\rightarrow$ $$ 2).
  * For Tree-based Models (Decision Tree, Random Forest, XGBoost) as they can handle these integers without assuming a false relationship.
* Pros:
  * Simple and creates no new columns.
* Cons:
  * Creates a false mathematical relationship. The model will think "Blue" (2) is greater than "Green" (1), which is meaningless and will confuse most models (like linear regression).

***

&#x20;<mark style="color:yellow;">**One-Hot Encoding**</mark>

* What it is: "Red" $$ $\rightarrow$ $$ `[1, 0, 0]`, "Green" $$ $\rightarrow$ $$ `[0, 1, 0]`, "Blue" $$ $\rightarrow$ $$ `[0, 0, 1]`
* When to Use:
  * The default, go-to for all nominal data (where order doesn't matter, e.g., "City," "Color").
  * Required for most models (Linear/Logistic Regression, SVM, k-NN, Neural Networks) to prevent false ordinal relationships.
* Pros:
  * Mathematically correct. It creates no false order.
* Cons:
  * Curse of Dimensionality: If you have a feature with 10,000 unique categories (e.g., "Zip Code"), this will create 10,000 new columns, which can make your dataset huge and slow to train.

***

## <mark style="color:blue;">4. Preprocessing Text Data (NLP)</mark>

<mark style="color:yellow;">**Cleaning & Normalization**</mark>

| **Technique**                                                                   | **When to Use**                                                                               | **Pros**                                                                                                   | **Cons**                                                                                                                        |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| <mark style="color:red;">Removing Stop Words</mark> ("a", "the", "is")          | Almost always, for vectorisation methods like BoW and TF-IDF, to reduce noise.                | Reduces the size of the vocabulary, speeds up training, and can improve model focus.                       | Can lose context. In sentiment analysis, "this movie is _not_ good" might become "movie good," completely flipping the meaning. |
| <mark style="color:red;">Stemming</mark> ("running" $$ $\rightarrow$ $$ "runn") | When you need a fast, crude way to normalize words. Good for large datasets or as a baseline. | Fast and computationally cheap.                                                                            | Aggressive and often inaccurate. It's just a rule-based "chopping" of words and can create non-existent words.                  |
| <mark style="color:red;">Lemmatization</mark> ("ran" $$ $\rightarrow$ $$ "run") | When you need accurate word normalization. This is the preferred method for most tasks.       | Linguistically accurate. It uses a dictionary to find the root form, so it's much "smarter" than stemming. | Slower than stemming because it needs to look up the word's context.                                                            |

***

<mark style="color:yellow;">**Vectorization (Turning Text to Numbers)**</mark>

| **Technique**                                      | **When to Use**                                                                                   | **Pros**                                                                                                                       | **Cons**                                                                                                                                         |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| <mark style="color:red;">Bag-of-Words (BoW)</mark> | As a simple, fast baseline for text classification (e.g., spam detection).                        | Simple to understand and implement.                                                                                            | Ignores word order ("man bit dog" = "dog bit man"). Ignores word importance (gives high scores to common, meaningless words).                    |
| <mark style="color:red;">TF-IDF</mark>             | A better default than BoW. Excellent for topic modeling, search engines, and text classification. | Smarter than BoW. It "finds" the important words by down-weighting common words (like "the") and boosting rare, topical words. | Still ignores word order and context. It doesn't understand the _meaning_ of words ("good" and "excellent" are treated as completely different). |

***

## <mark style="color:blue;">5. Preprocessing Image Data (CV)</mark>

<mark style="color:yellow;">**Resizing**</mark>

* When to Use: This is mandatory for all models. Every image in a batch must be the exact same height and width (e.g., 224x224) to be fed into the network.
* Pros:
  * Enables batch processing.
  * Reduces memory usage and speeds up training.
* Cons:
  * Can distort the image if the aspect ratio is changed (e.g., squashing a wide image into a square).
  * Can lose information if you resize a large, detailed image to be very small.

***

<mark style="color:yellow;">**Pixel Scaling**</mark>

* When to Use: This is also mandatory. Neural networks train poorly on pixel values in the `[0-255]` range.

| **Method**                                                             | **When to Use**                                                                      | **Pros**                                                                                                                      | **Cons**                                                                                     |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| <mark style="color:red;">Normalization</mark> `[0, 1]` (Divide by 255) | The simplest, most common method. A great default for training a model from scratch. | Simple and effective. Puts all pixels in a clean, consistent range.                                                           | Less common for transfer learning.                                                           |
| <mark style="color:red;">Standardization</mark> (ImageNet Stats)       | When using Transfer Learning (e.g., a pre-trained ResNet, VGG, or EfficientNet).     | Matches the pre-training conditions. These models _expect_ data to be normalised with a specific mean and standard deviation. | You must use the _exact_ mean/std values the model was trained on, or performance will drop. |

***

<mark style="color:yellow;">**Data Augmentation**</mark>

* When to Use: Always, during training, especially when you have a small dataset.
* Pros:
  * Acts as a regulariser. Prevents overfitting by creating "new" training images.
  * Makes the model more robust. A model trained on random rotations will learn to recognise an object even if it's upside-down.
* Cons:
  * Increases training time.
  * If augmentations are too "aggressive" (e.g., too much rotation or color shift), the model may fail to learn the real features.

