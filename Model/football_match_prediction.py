# -*- coding: utf-8 -*-
"""
Football Match Prediction Model using Spanish Super League Dataset

This script performs exploratory data analysis and trains various machine learning 
models to predict match outcomes (home win, draw, away win) based on the provided dataset.
"""

# Importing necessary libraries
import pandas as pd
import numpy as np

# For plots
import matplotlib.pyplot as plt
import seaborn as sns

# For machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data_path = 'C:/Users/sravan kumar Bari/OneDrive/Desktop/Football Match Prediction/Dataset/FMEL_Dataset.csv'  # Update path if needed
data = pd.read_csv(data_path)

# Initial info
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())

# Create target variable: match_result
# 0 = draw, 1 = home win, 2 = away win
def match_result(row):
    if row['localGoals'] > row['visitorGoals']:
        return 1
    elif row['localGoals'] < row['visitorGoals']:
        return 2
    else:
        return 0

data['match_result'] = data.apply(match_result, axis=1)

# Encode categorical variables: localTeam, visitorTeam, season (optional)
# For simplicity, encode team names as numbers
le_local = LabelEncoder()
le_visitor = LabelEncoder()

data['localTeam_enc'] = le_local.fit_transform(data['localTeam'])
data['visitorTeam_enc'] = le_visitor.fit_transform(data['visitorTeam'])

# Features and target selection
# Features: localTeam_enc, visitorTeam_enc, division, round (encoded or numeric), maybe season encoded
# Encode 'division' and 'season' categorical variables
le_division = LabelEncoder()
le_season = LabelEncoder()

data['division_enc'] = le_division.fit_transform(data['division'])
data['season_enc'] = le_season.fit_transform(data['season'])

features = ['localTeam_enc', 'visitorTeam_enc', 'division_enc', 'round', 'season_enc']
X = data[features]
y = data['match_result']

# Print dataset stats
print("\nMatch result distribution:")
print(y.value_counts(normalize=True))

# Split into training and testing dataset (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Function to train and evaluate model
def train_evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)
    print(f"{model_name} Accuracy: {score:.4f}")
    return score

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gaussian NB': GaussianNB(),
    'Support Vector Machine': SVC(random_state=42)
}

# Train and evaluate all models
results = {}
for name, model in models.items():
    score = train_evaluate_model(model, name)
    results[name] = score

# Plotting Match Result distribution
plt.figure(figsize=(8,6))
sns.countplot(x='match_result', data=data, palette='viridis')
plt.xticks(ticks=[0,1,2], labels=['Draw', 'Home Win', 'Away Win'])
plt.title('Distribution of Match Results')
plt.xlabel('Match Result')
plt.ylabel('Number of Matches')
plt.show()

# Plot winning percentages by season for home win, draw, away win
seasonly = data.groupby('season')[['match_result']].apply(
    lambda x: pd.Series({
        'Home Win %': np.mean(x == 1),
        'Draw %': np.mean(x == 0),
        'Away Win %': np.mean(x == 2)
    })
).reset_index()

plt.figure(figsize=(20,6))
plt.plot(seasonly['season'], seasonly['Home Win %'], label='Home Win %', color='green')
plt.plot(seasonly['season'], seasonly['Draw %'], label='Draw %', color='blue')
plt.plot(seasonly['season'], seasonly['Away Win %'], label='Away Win %', color='red')
plt.xticks(rotation=90)
plt.title('Match Result Percentages Over Seasons')
plt.xlabel('Season')
plt.ylabel('Percentage')
plt.legend()
plt.grid(True)
plt.show()

# Conclusion printout
best_model_name = max(results, key=results.get)
print(f"\nBest performing model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# End of script

