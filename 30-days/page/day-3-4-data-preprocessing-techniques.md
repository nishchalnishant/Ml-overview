# Day 3-4: Data Preprocessing Techniques:

Here are the detailed notes for **Day 3-4: Data Preprocessing Techniques**:

#### **Introduction to Data Preprocessing**

* **Definition**: Data preprocessing is the process of transforming raw data into a clean and usable format. This step is essential because real-world data is often incomplete, noisy, and inconsistent.
* **Importance**: Properly preprocessing data ensures that machine learning models are more accurate, efficient, and interpretable.

#### **1. Data Cleaning**

Data cleaning involves identifying and correcting (or removing) inaccurate records from the dataset. This step ensures the dataset is free from errors that could skew model performance.

**Handling Missing Values:**

* **Why are there missing values?**: Data might be missing due to human errors, system glitches, or missing information during data collection.
* **Common Strategies for Handling Missing Data**:
  * **Remove Missing Data**:
    * **Row Deletion**: Remove rows with missing values if they are relatively few. This works well when missing data is minimal and random.
      * Example: `df.dropna()`
    * **Column Deletion**: Remove columns with too many missing values. Only do this when the column is not critical.
      * Example: `df.drop(columns=['column_name'])`
  * **Imputation**:
    * **Mean/Median/Mode Imputation**: Replace missing values with the mean, median, or mode of the feature.
      * **Mean**: Useful for continuous features with a normal distribution.
      * **Median**: Used when the feature has outliers.
      * **Mode**: For categorical data.
      * Example: `df['column'].fillna(df['column'].mean())`
    * **Forward Fill (ffill)** and **Backward Fill (bfill)**: Fill missing values using the previous or next observation.
      * Example: `df.fillna(method='ffill')`
    * **Model-based Imputation**: Use machine learning algorithms to predict missing values based on other features in the dataset.

**Handling Outliers:**

* **Why are outliers a problem?**: Outliers can distort statistical analyses and affect the performance of machine learning models, especially those sensitive to extremes (e.g., linear regression).
* **Common Techniques**:
  * **Z-Score Method**: If the Z-score (standardized value) of a data point is beyond a certain threshold (typically 3), it can be considered an outlier.
    * Formula: ( Z = \frac{(X - \mu)}{\sigma} )
      * ( X ): Data point
      * ( \mu ): Mean of the data
      * ( \sigma ): Standard deviation
    * Example: `from scipy import stats; z = np.abs(stats.zscore(df['column']))`
  * **IQR Method**: Use the interquartile range (IQR) to detect outliers. Points outside 1.5 times the IQR above the third quartile (Q3) or below the first quartile (Q1) are considered outliers.
    * Formula: ( IQR = Q3 - Q1 )
      * Example: `Q1 = df['column'].quantile(0.25); Q3 = df['column'].quantile(0.75); IQR = Q3 - Q1`

#### **2. Feature Scaling and Normalization**

Machine learning algorithms can be sensitive to the scale of the data, especially distance-based algorithms (e.g., K-Nearest Neighbors, Support Vector Machines, etc.). Feature scaling and normalization ensure all features are on a similar scale, improving model performance.

**Standardization (Z-score Normalization):**

* **Purpose**: Rescales data to have a mean of 0 and a standard deviation of 1.
  * **Formula**: ( Z = \frac{X - \mu}{\sigma} )
    * ( X ): Original data point
    * ( \mu ): Mean of the feature
    * ( \sigma ): Standard deviation of the feature
  * Example: `from sklearn.preprocessing import StandardScaler; scaler = StandardScaler(); df_scaled = scaler.fit_transform(df)`
* **When to use**:
  * When features follow a normal distribution or when algorithms assume standard normally distributed data (e.g., SVM, logistic regression).

**Min-Max Scaling:**

* **Purpose**: Rescales data to a fixed range, typically \[0, 1].
  * **Formula**: ( X\_{scaled} = \frac{X - X\_{min\}}{X\_{max} - X\_{min\}} )
    * ( X\_{min} ) and ( X\_{max} ) are the minimum and maximum values of the feature, respectively.
  * Example: `from sklearn.preprocessing import MinMaxScaler; scaler = MinMaxScaler(); df_scaled = scaler.fit_transform(df)`
* **When to use**:
  * When the data does not follow a Gaussian distribution or when algorithms do not assume normality (e.g., KNN, neural networks).

**MaxAbs Scaling:**

* **Purpose**: Scales data by dividing each feature by its maximum absolute value, preserving sparsity.
  * **Formula**: ( X\_{scaled} = \frac{X}{|X\_{max}|} )
  * Example: `from sklearn.preprocessing import MaxAbsScaler; scaler = MaxAbsScaler(); df_scaled = scaler.fit_transform(df)`
* **When to use**:
  * For data that is sparse (many zeros) and when preserving zero entries is crucial (e.g., text data in bag-of-words format).

**Robust Scaling:**

* **Purpose**: Rescales data using the median and IQR, making it less sensitive to outliers.
  * **Formula**: ( X\_{scaled} = \frac{X - X\_{median\}}{IQR} )
  * Example: `from sklearn.preprocessing import RobustScaler; scaler = RobustScaler(); df_scaled = scaler.fit_transform(df)`
* **When to use**:
  * When your data contains significant outliers.

#### **3. Encoding Categorical Variables**

Categorical features (like "Country" or "Color") need to be converted into numerical formats because machine learning models usually work with numbers.

**One-Hot Encoding:**

* **Purpose**: Converts categorical variables into binary vectors (0/1). Each category becomes a separate binary feature.
  * Example: If the "Color" feature has three categories \[Red, Blue, Green], it will be transformed into three binary features: Red (1/0), Blue (1/0), Green (1/0).
  * Example: `pd.get_dummies(df, columns=['column_name'])`
* **When to use**:
  * When categorical variables are nominal (no inherent order, like “Color”).

**Label Encoding:**

* **Purpose**: Assigns a unique integer to each category.
  * Example: Red -> 1, Blue -> 2, Green -> 3.
  * Example: `from sklearn.preprocessing import LabelEncoder; encoder = LabelEncoder(); df['column_name'] = encoder.fit_transform(df['column_name'])`
* **When to use**:
  * When categorical variables are ordinal (have a meaningful order, like “Size”: Small, Medium, Large).

**Target Encoding:**

* **Purpose**: Replaces categories with the mean of the target variable for that category.
  * Example: If the “Color” feature is Red, Blue, and Green and the target is sales, you can replace “Red” with the average sales value for Red.
  * This method can improve performance but may cause data leakage if not done correctly.

#### **4. Feature Engineering**

Feature engineering is the process of creating new features from raw data to improve model performance.

* **Polynomial Features**: Creating interaction terms and polynomial terms for continuous features.
  * Example: ( x\_1^2, x\_2^3, x\_1 \times x\_2 )
  * Example: `from sklearn.preprocessing import PolynomialFeatures; poly = PolynomialFeatures(degree=2); df_poly = poly.fit_transform(df)`
* **Binning**: Converting continuous features into categorical ones by dividing them into bins (intervals).
  * Example: Age groups like “0-18”, “19-35”, “36-60”, etc.
  * Example: `pd.cut(df['Age'], bins=[0, 18, 35, 60, 100])`

#### Summary for Day 3-4:

* **Data Cleaning**:
  * Handle missing values by removing them or imputing (mean, median, mode).
  * Detect and handle outliers using Z-score or IQR methods.
* **Feature Scaling**:
  * Standardize features using Z-score or scale them to a specific range (Min-Max scaling).
  * Use Robust Scaling when dealing with outliers.
* **Encoding Categorical Variables**:
  * Use One-Hot Encoding for nominal categories.
  * Use Label Encoding for ordinal categories.
* **Feature Engineering**:
  * Create new features or modify existing ones to enhance model performance.

These preprocessing steps are essential for ensuring that your machine learning models are robust, efficient, and effective.
