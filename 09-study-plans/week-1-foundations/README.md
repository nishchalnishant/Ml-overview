---
module: Study Plans
topic: Week 1 (Days 1-7): Foundations
subtopic: ""
status: unread
tags: [studyplans, ml, week-1-days-1-7-foundations]
---
# Week 1 (Days 1-7): Foundations

**Goal:** Build a solid grounding in ML theory, data handling, and statistics before touching any algorithms. These are the topics that trip up candidates who skip straight to models.

---

## What This Week Covers

| Days   | Topic                              | Key Concepts                                              |
|--------|------------------------------------|-----------------------------------------------------------|
| 1-2    | Intro to ML & Core Principles      | Bias-variance tradeoff, No Free Lunch, Occam's Razor      |
| 3-4    | Data Preprocessing & Feature Eng.  | Data leakage, imputation (MICE/indicator), scaling        |
| 5-7    | EDA, Statistics & Data Quality     | Distribution checks, outlier detection, skew, P-values, CLT, Bayes' theorem |

---

## Focus Areas

- **Bias-Variance Tradeoff:** Understand it mathematically, not just intuitively. Expect derivations.
- **Data Leakage:** The most common pipeline bug in interviews. Know how to detect and prevent it at every stage: imputation, scaling, encoding, and cross-validation.
- **Imputation strategies:** MICE vs. simple mean/median vs. indicator variables — know when each applies and understand MCAR vs. MAR vs. MNAR.
- **EDA methodology:** Don't just run plots — form hypotheses before plotting, then check whether plots confirm or contradict them.
- **Statistical testing:** Be able to explain p-values without the textbook definition. Know when to use t-test vs. chi-squared vs. ANOVA. Interviewers probe this hard.
- **Probability fundamentals:** Conditional probability, Bayes, and the law of total expectation are frequent whiteboard topics.

---

## Daily Study Pattern

1. Read the linked material (30-45 min).
2. Write a 3-sentence summary of the core tradeoff or mechanism in your own words.
3. Answer at least two interview-style questions from the section before moving on.
4. Write one concrete code snippet implementing the main concept (even if just 5 lines).

---

## Linked Resources

- [Fundamentals of Machine Learning](../../07-interview-prep/ml/fundamentals-of-machine-learning.md)
- [Data Preprocessing & Feature Engineering](../../07-interview-prep/ml/data-preprocessing-and-feature-engineering.md)
- [Probability & Statistics](../../07-interview-prep/ml/probability-and-statistics.md)
- [AI & ML Revision Guide (night-before cheat sheet)](../../01-foundations/01-ai-ml-systems-and-application.md)
- Day files in this folder: day-1-2, day-3-4, day-5-7

---

## Project for This Week

**Dataset:** Use any Kaggle tabular dataset (e.g., Titanic, House Prices, or Credit Card Fraud).

**Deliverable:** A documented Python notebook covering:
1. EDA: distributions, missingness heatmap, correlation matrix, 3 observations
2. Preprocessing pipeline: imputation + scaling + encoding — all fit on train only
3. A baseline logistic regression with proper 5-fold cross-validation
4. One paragraph explaining: what data leakage risk you identified and how you prevented it

---

## End-of-Week Check

- Can you explain the bias-variance tradeoff and where it shows up in regularization?
- Can you walk through a full data cleaning pipeline for a messy tabular dataset without any leakage?
- Can you derive or explain Bayes' theorem from first principles?
- Do you know when to use a t-test vs. a chi-squared test vs. ANOVA?
- Can you explain the difference between MCAR, MAR, and MNAR and what each implies for imputation strategy?
