# ============================================================
# User Behaviour Analysis for Optimizing Engagement on Social Media
# Course: Fundamentals of Data Analytics
# Author: Oyewole Oladimeji Abdulqudus | Q1045935
# Berlin School of Business and Innovation | 2025
# ============================================================

# ---- PHASE 1: DATA LOADING AND EXPLORATION ----

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/content/complete_social_media_user_data.csv')

# Preview the dataset
print(df)

# ---- PHASE 2: MISSING VALUES AND OUTLIER DETECTION ----

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Fill missing values with column mean (if any)
df.fillna(df.mean(), inplace=True)

# Descriptive statistics
print("Descriptive statistics of dataset:")
print(df.describe())

# Identify outliers using Interquartile Range (IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

print("Number of outliers per column:")
print(outliers)

# Visualise outliers using boxplots
plt.figure(figsize=(8, 4))
sns.boxplot(data=df[["Posts", "Likes", "Comments", "Shares", "Engagement_Score"]])
plt.title("Outlier Detection in User Activity Data")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# ---- PHASE 3: DATA VISUALISATION ----

# Distribution of time spent on the platform
plt.figure(figsize=(10, 6))
sns.histplot(df['Time_Spent_Hours'], bins=15, kde=False, color='green')
plt.title('Distribution of Time Spent on the Platform')
plt.xlabel('Time Spent (Hours)')
plt.ylabel('Number of Users')
plt.show()

# Average user activity: Posts, Likes, Comments, Shares
activity_columns = ["Posts", "Likes", "Comments", "Shares"]
average_activity = df[activity_columns].mean()

plt.figure(figsize=(8, 5))
average_activity.plot(kind="bar", alpha=0.7)
plt.xlabel("Activity Type")
plt.ylabel("Average Count")
plt.title("Average User Activity on Social Media")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.show()

# User Engagement Score over Profile Age
plt.figure(figsize=(8, 4))
sns.lineplot(x=df["Profile_Age_Days"], y=df["Engagement_Score"])
plt.title("User Engagement Score Over Profile Age")
plt.xlabel("Profile Age (Days)")
plt.ylabel("Engagement Score")
plt.show()

# ---- PHASE 4: CORRELATION ANALYSIS ----

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ---- PHASE 5: CLUSTERING ----

# 5.1 K-Means Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Feature selection for clustering
features = df[['Likes', 'Comments', 'Shares', 'Messages_Sent', 'Time_Spent_Hours']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster_kmeans'] = kmeans.fit_predict(scaled_features)

# Display cluster distribution
print(df['cluster_kmeans'].value_counts())

# 5.2 DBSCAN Clustering
from sklearn.cluster import DBSCAN

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['cluster_dbscan'] = dbscan.fit_predict(scaled_features)

# Identify outliers (labelled as -1 by DBSCAN)
print(df[df['cluster_dbscan'] == -1])
