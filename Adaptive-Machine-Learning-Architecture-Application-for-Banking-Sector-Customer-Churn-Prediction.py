import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import lightgbm as lgb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
import random
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import BorderlineSMOTE

df  = pd.read_csv('Bank_churn_prediction.csv')

df = df.drop(columns=['customer_id'])

# Ordinal encoding for nominal variables
nominal_columns = df.select_dtypes(include=['object', 'category']).columns
ordinal_encoder = OrdinalEncoder()
df[nominal_columns] = ordinal_encoder.fit_transform(df[nominal_columns])

# Replace numerical missing values with interpolation
imputer = SimpleImputer(strategy='median')
df[df.select_dtypes(include=['float64', 'int64']).columns] = \
imputer.fit_transform(df[df.select_dtypes(include=['float64', 'int64']).columns])

# Treat missing values of categorical attributes as a separate category
df[nominal_columns] = df[nominal_columns].fillna(-1)


# separate features and labels
X_data = df.drop(['churn'] , axis = 1)
y = df['churn']

BSMOTE = BorderlineSMOTE()
X_resampled, y_resampled = BSMOTE.fit_resample(X_data,y)

df_resampled = pd.DataFrame(X_resampled, columns=X_data.columns)

# Add the resampled target variable to the DataFrame
df_resampled['churn'] = y_resampled
df_resampled

features = df_resampled.drop(['churn'] , axis = 1)
churn = df_resampled['churn']
X_train, X_test, y_train, y_test = train_test_split(features, churn, test_size=0.3, random_state=0)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class Spider:
    def __init__(self, position, fitness):
        self.position = position
        self.fitness = fitness

def initialize_population(search_space, num_spiders):
    population = []
    for _ in range(num_spiders):
        position = np.random.randint(2, size=search_space)
        fitness = evaluate_fitness(position)
        population.append(Spider(position, fitness))
    return population

def update_position(spider, best_neighbor, transfer_func):
    difference = spider.position - best_neighbor.position
    update_value = transfer_func(difference)
    if np.random.rand() < 0.5:
        spider.position = np.where(spider.position != best_neighbor.position, update_value, spider.position)

def transfer_function(difference):
    return np.where(difference != 0, np.random.randint(2, size=difference.shape), difference)

def evaluate_fitness(position):
    selected_features = [i for i in range(len(position)) if position[i] == 1]
    if not selected_features:
        return 0  # If no features are selected, return 0 fitness

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, churn, test_size=0.1, random_state=42)

    # Train a model (e.g., Random Forest) and evaluate its performance
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    fitness = accuracy_score(y_test, y_pred)  # Use accuracy as the fitness value

    return fitness

def L_BSSA(search_space, num_spiders, max_iterations):
    population = initialize_population(search_space, num_spiders)
    for _ in range(max_iterations):
        best_spider = max(population, key=lambda spider: spider.fitness)
        for spider in population:
            neighbors = np.random.choice(population, size=2, replace=False)
            best_neighbor = max(neighbors, key=lambda neighbor: neighbor.fitness)
            update_position(spider, best_neighbor, transfer_function)
        for spider in population:
            spider.fitness = evaluate_fitness(spider.position)
    return max(population, key=lambda spider: spider.fitness)



# Assuming the last column is the target variable
features = df_resampled.drop(['churn'] , axis = 1)
churn = df_resampled['churn']
column_names = features.columns[:-1].tolist()

search_space = features.shape[1]  # Number of features
num_spiders = 7
max_iterations = 12

best_spider = L_BSSA(search_space, num_spiders, max_iterations)

# Get indices of the features selected by the best spider
selected_features = np.where(best_spider.position == 1)[0]
# Get the names of the top selected features
top_features = [column_names[i] for i in selected_features]

print("Best solution found:", best_spider.position)
print("Best fitness:", best_spider.fitness)
print("Top selected features:", top_features)

selected_df = features[top_features]

X_train, X_test, y_train, y_test = train_test_split(selected_df, churn, test_size=0.3, random_state=0)
stratify = y_train

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, auc #Added roc_curve and auc
import matplotlib.pyplot as plt #Added matplotlib

# Assuming you have X_train, y_train, X_test, y_test defined somewhere

# Step 1: Count samples in each class
class_counts = np.bincount(y_train)
total_samples = len(y_train)

# Step 2: Compute class weights (inverse of class frequencies)
class_weights = {cls: total_samples / count for cls, count in enumerate(class_counts)}

# Step 3: Normalize weights (optional, to keep weights proportional)
total_weight = sum(class_weights.values())
class_weights = {cls: weight / total_weight for cls, weight in class_weights.items()}

print("Class weights:", class_weights)

# Step 4: Pass to Random Forest
rf_classifier = RandomForestClassifier(class_weight=class_weights, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate precision
precision = precision_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate f1_score
f1Score = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("f1_score:", f1Score)

# Generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    rf_classifier, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)
)

# Calculate mean and standard deviation of training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot ROC curve
y_proba = rf_classifier.predict_proba(X_test)[:, 1]  # Probabilities of the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_proba) # roc_curve is defined here
roc_auc = auc(fpr, tpr) # auc is defined here

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()