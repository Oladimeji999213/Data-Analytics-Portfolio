# Social Media User Behaviour Analysis

**Course:** Fundamentals of Data Analytics
**Institution:** Berlin School of Business and Innovation
**Year:** 2025
**Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

-----

## Project Overview

This project analyses user behaviour data from a social media platform to identify distinct engagement patterns. Using clustering algorithms, users are segmented based on their activity levels to derive actionable insights for content personalisation, targeted marketing, and anomaly detection.

-----

## Dataset

The dataset contains 100 user records with the following features:

- User_ID, Posts, Likes, Comments, Shares
- Messages_Sent, Messages_Received
- Time_Spent_Hours
- Reaction_to_Recommendations
- Follower_Count, Following_Count
- Profile_Age_Days
- Engagement_Score

-----

## Methodology

**1. Data Exploration**

- Descriptive statistics to understand data distribution
- Checked for missing values (none found)
- Outlier detection using IQR and boxplots

**2. Data Visualisation**

- Distribution of time spent on platform
- Average user activity across Posts, Likes, Comments, Shares
- Engagement Score trend over Profile Age
- Correlation heatmap across all features

**3. Key Correlations Found**

- Likes vs Engagement Score: 0.95 (strongest predictor)
- Comments vs Engagement Score: 0.87
- Shares vs Engagement Score: 0.80
- Follower Count vs Engagement Score: 0.67

**4. Clustering**

- K-Means Clustering (3 clusters) for user segmentation
- DBSCAN for anomaly and bot detection

-----

## Key Findings

- Users naturally fall into three engagement groups: high, medium, and low activity
- Likes is the single strongest driver of engagement score
- DBSCAN successfully flagged unusual activity patterns that could indicate bot accounts
- Users interact with content far more than they create it

-----

## Files

|File                                |Description                                               |
|------------------------------------|----------------------------------------------------------|
|`social_media_behaviour_analysis.py`|Full Python script with EDA, visualisation, and clustering|

-----

## Skills Demonstrated

Exploratory Data Analysis, Data Cleaning, Statistical Analysis, K-Means Clustering, DBSCAN, Data Visualisation, Python, Scikit-learn, Seaborn
