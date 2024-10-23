# Day 5-7: Exploratory Data Analysis (EDA)

Here are the detailed notes for **Day 5-7: Exploratory Data Analysis (EDA)** from your preparation schedule:

#### **Introduction to Exploratory Data Analysis (EDA)**

* **Definition**: EDA is the process of summarizing the main characteristics of a dataset through visualizations and statistical techniques before applying any machine learning models. It helps to understand patterns, detect anomalies, and identify relationships between variables.
* **Importance**: EDA helps in uncovering underlying patterns, cleaning data, validating assumptions, and selecting appropriate machine learning models.

#### **Steps in EDA**

1. **Data Overview**
   * **Understand Data Structure**: Start by examining the size, shape, and data types of the dataset.
     * `df.shape`: Provides the number of rows and columns.
     * `df.info()`: Displays data types and non-null counts.
     * `df.describe()`: Summarizes numerical features (mean, std, min, max, quartiles).
2. **Identify Missing Data**:
   * **Missing Data Detection**:
     * `df.isnull().sum()`: Detects missing values in each column.
   * **Visualization**:
     * Use a heatmap or bar chart to visualize missing data patterns.
     * Example: `sns.heatmap(df.isnull(), cbar=False)`.

#### **1. Visualization Techniques**

Visualizing data is a crucial part of EDA as it reveals patterns, distributions, correlations, and outliers in a dataset.

**1.1 Univariate Analysis (Single Feature):**

* Focuses on understanding the distribution and characteristics of a single feature (variable).

**For Numerical Data:**

* **Histograms**:
  * **Purpose**: Show the distribution of a numerical feature.
  * **Interpretation**: Check if the data is normally distributed, skewed, or contains outliers.
  * Example: `df['column_name'].hist(bins=30, figsize=(8,6))`
  * **Key Observations**: Look for normality, skewness, and any extreme values (outliers).
* **Boxplots**:
  * **Purpose**: Display the summary statistics (median, quartiles) and highlight outliers.
  * Example: `sns.boxplot(df['column_name'])`
  * **Key Observations**: Identify central tendencies and outliers using whiskers.
* **KDE Plot (Kernel Density Estimation)**:
  * **Purpose**: Show the probability density of the feature, providing a smoothed distribution curve.
  * Example: `sns.kdeplot(df['column_name'])`
  * **Key Observations**: Examine the shape of the distribution.

**For Categorical Data:**

* **Bar Plots**:
  * **Purpose**: Show the frequency or proportion of categories within a categorical feature.
  * Example: `df['column_name'].value_counts().plot(kind='bar', figsize=(8,6))`
  * **Key Observations**: Identify dominant categories or imbalances in class distribution.

**1.2 Bivariate Analysis (Two Features):**

* Focuses on understanding the relationship between two variables.

**For Numerical-Numerical Data:**

* **Scatter Plots**:
  * **Purpose**: Show the relationship between two numerical features.
  * Example: `plt.scatter(df['feature1'], df['feature2'])`
  * **Key Observations**: Look for linear or nonlinear relationships, clusters, or outliers.
* **Correlation Matrix**:
  * **Purpose**: Measure the linear correlation between numerical features (values between -1 and 1).
  * Example: `df.corr()`
  * **Key Observations**: Use heatmaps to visualize the strength of correlation.
    * Example: `sns.heatmap(df.corr(), annot=True, cmap='coolwarm')`
  * **Key Metrics**:
    * **Positive Correlation (Close to 1)**: As one variable increases, the other tends to increase.
    * **Negative Correlation (Close to -1)**: As one variable increases, the other tends to decrease.
    * **Zero Correlation**: No linear relationship.

**For Numerical-Categorical Data:**

* **Box Plots**:
  * **Purpose**: Compare the distribution of a numerical feature across different categories.
  * Example: `sns.boxplot(x='categorical_feature', y='numerical_feature', data=df)`
  * **Key Observations**: Identify differences in central tendencies and variability among categories.
* **Violin Plots**:
  * **Purpose**: Show the distribution of numerical data for each category and its density.
  * Example: `sns.violinplot(x='categorical_feature', y='numerical_feature', data=df)`
  * **Key Observations**: Compare distributions while preserving more details about density than a box plot.
* **Count Plots** (for categorical data):
  * **Purpose**: Display counts of observations for each category.
  * Example: `sns.countplot(x='categorical_feature', data=df)`
  * **Key Observations**: Spot any significant class imbalances.

**For Categorical-Categorical Data:**

* **Cross Tabulation (Contingency Table)**:
  * **Purpose**: Summarize the relationship between two categorical variables.
  * Example: `pd.crosstab(df['categorical1'], df['categorical2'])`
  * **Key Observations**: Identify patterns or dependencies between categories.
* **Stacked Bar Charts**:
  * **Purpose**: Visualize the proportion of one categorical variable across different categories of another variable.
  * Example: Use `sns.countplot` and `hue` parameter to add a second category: `sns.countplot(x='categorical1', hue='categorical2', data=df)`.

#### **2. Multivariate Analysis**

Multivariate analysis examines the relationships between three or more variables, helping to identify complex interactions.

**Pair Plots:**

* **Purpose**: Visualize pairwise relationships between numerical variables.
* **Interpretation**: Scatter plots on the off-diagonal show relationships, and histograms on the diagonal show distributions of each feature.
* Example: `sns.pairplot(df)`
* **Key Observations**: Identify correlations and patterns across multiple features simultaneously.

**Heatmap for Multivariate Correlation:**

* **Purpose**: Visualize the correlation matrix in a more informative and color-coded format.
* **Interpretation**: Easily detect strong correlations or multicollinearity (strong correlations between independent variables).
* Example: `sns.heatmap(df.corr(), annot=True, cmap='viridis')`
* **Key Observations**: Features that are highly correlated might cause multicollinearity, and you might want to drop one of them.

**Pivot Tables:**

* **Purpose**: Summarize data across multiple dimensions.
* **Example**: `pd.pivot_table(df, values='target', index='feature1', columns='feature2', aggfunc=np.mean)`
* **Key Observations**: Identify interactions between categorical and numerical features.

#### **3. Outlier Detection and Handling**

Outliers are extreme values that differ significantly from the majority of the data. They can distort results, so it's essential to identify and decide how to handle them.

**Detection Techniques:**

* **Box Plots**: The whiskers in box plots often indicate the range within which most data points lie, and points outside the whiskers are considered outliers.
  * Example: `sns.boxplot(df['column_name'])`
* **Z-Score**: Standardize the data and flag any points with Z-scores greater than 3 (or less than -3) as outliers.
  * Formula: ( Z = \frac{(X - \mu)}{\sigma} )
* **IQR Method**: Flag data points outside 1.5 times the interquartile range as outliers.
  * Formula: ( IQR = Q3 - Q1 )

**Handling Outliers:**

* **Removing Outliers**: In some cases, simply removing outliers might be the best approach.
* **Transformation**: Apply transformations (e.g., logarithmic or square root) to mitigate the effect of outliers.
* **Imputation**: Replace outliers with the mean, median, or a value closer to the main distribution.

#### **4. Feature Relationships and Insights**

Once patterns are identified through visualization and correlation, it's essential to derive insights about how features relate to the target variable and to each other.

* **Feature Importance**: Understand which features are likely to have the strongest impact on your target variable by looking at correlations or using tree-based methods (like Random Forests or Decision Trees) to rank feature importance.
* **Multicollinearity**: High correlations between independent variables can distort the performance of some models (like linear regression). Use the correlation matrix and Variance Inflation Factor (VIF) to detect multicollinearity.

#### **Tools for EDA**:

1. **Pandas**:
   * Best for tabular data manipulation and summary statistics.
   * Example: `df.describe()`, `df.info()`, `df.groupby()`
2. **Matplotlib**:
   * Best for basic plotting.
   * Example: `plt.hist(df['column_name'])`
3. **Seaborn**:
   * Built on top of Matplotlib, Seaborn provides more aesthetically pleasing plots and is easier to use for complex visualizations.
   * Example: `sns.heatmap(df.corr())`
4. **Plotly**:
   * Great for creating interactive plots.
   * Example: `plotly.express.scatter(df, x='feature1', y='feature2')`

#### Summary for Day 5

\-7:

* **Data Overview**: Understand the structure, data types, and summary statistics of your dataset.
* **Univariate Analysis**: Explore individual features, their distributions, and outliers.
* **Bivariate Analysis**: Explore the relationships between pairs of variables.
* **Multivariate Analysis**: Use techniques like pair plots, heatmaps, and pivot tables to understand interactions between multiple variables.
* **Visualization**: Use tools like Matplotlib and Seaborn for effective visual analysis.
* **Outlier Detection**: Identify and decide how to handle outliers, using techniques like Z-score, IQR, and box plots.

By the end of Day 5-7, you should have a thorough understanding of the dataset and be ready to apply machine learning algorithms with greater confidence.
