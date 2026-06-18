import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 1. Load Data & Initial Inspection
df = pd.read_csv("titanic.csv")

print("Data Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# 2. Summary Statistics
print("\nSummary Stats:\n", df.describe(include='all'))

# 3. Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Distribution Analysis

# Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Fare by Class
plt.figure(figsize=(8, 5))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare by Class')
plt.show()

# Survival by Gender
plt.figure(figsize=(8, 4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Cross Tabulation
print("\nSurvival Rate by Passenger Class:")
print(pd.crosstab(df['Pclass'], df['Survived'], normalize='index') * 100)

# Outlier Detection
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Fare'])
plt.title('Fare Outliers')
plt.show()

z_scores = np.abs(stats.zscore(df['Fare']))
outliers = df[z_scores > 3]

print(f"\nFound {len(outliers)} fare outliers")

# Advanced Visualization - Facet Grid
g = sns.FacetGrid(df, col='Survived', row='Pclass', height=3)
g.map(sns.histplot, 'Age', bins=20)

# Pairplot
sns.pairplot(
    df[['Age', 'Fare', 'Parch', 'Survived']],
    hue='Survived'
)

plt.show()

print("\nEDA Completed Successfully!")
