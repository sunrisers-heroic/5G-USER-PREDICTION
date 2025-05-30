# -*- coding: utf-8 -*-
"""5G.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15JHixwPR-jXOKtvMKJK_2wW8213LFrPW

# Importing Libraries
"""

# General Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Scikit-learn Libraries
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    confusion_matrix,
    accuracy_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# XGBoost Library
import xgboost as xgb

# Imbalanced Data Handling
from imblearn.over_sampling import SMOTE

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

"""# DataSet

1. Import Dataset
"""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("liukunxin/dataset")

print("Path to dataset files:", path)

"""Load Dataset"""

train = pd.read_csv("/root/.cache/kagglehub/datasets/liukunxin/dataset/versions/1/train.csv")
test = pd.read_csv("/root/.cache/kagglehub/datasets/liukunxin/dataset/versions/1/test.csv")
sample = pd.read_csv("/root/.cache/kagglehub/datasets/liukunxin/dataset/versions/1/sample.csv")

train = train.sample(n=20000, random_state=42)

"""Data Cleaning

1 . Removing Days Not Useful
"""

days=['active_days01', 'active_days02',
       'active_days03', 'active_days04', 'active_days05', 'active_days06',
       'active_days07', 'active_days08', 'active_days09', 'active_days10',
       'active_days11', 'active_days12', 'active_days13', 'active_days14',
       'active_days15', 'active_days16', 'active_days17', 'active_days18',
       'active_days19', 'active_days20', 'active_days21', 'active_days22',
       'active_days23']
train.drop(columns=days,inplace=True)
test.drop(columns=days,inplace=True)

"""2 . Removing User id and Area ID"""

train.drop(columns=['user_id', 'area_id'], inplace=True)
test.drop(columns=['user_id', 'area_id'], inplace=True)

"""3 . Code"""

train.info()

"""# Training and Testing Data

1 .Label Encoding
"""

le = LabelEncoder()
train['is_5g'] = le.fit_transform(train['is_5g'])
sample['is_5g'] = le.transform(sample['is_5g'])

"""2. Splitting Train and Test Data"""

x_train = train.drop(columns=['is_5g']).values.astype(np.float32)
y_train = train['is_5g'].values.astype(np.float32)
x_test = test.values.astype(np.float32)
y_test = sample['is_5g'].values.astype(np.float32)

"""3.Converting To Tensors"""

x_train_tensor = torch.tensor(x_train)
y_train_tensor = torch.tensor(y_train)
x_test_tensor = torch.tensor(x_test)
y_test_tensor = torch.tensor(y_test)

"""# Sci-Kit Traditional Method

Train Test Split
"""

x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

"""Model Creation"""

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_balanced)
x_validation_scaled = scaler.transform(x_validation)
x_test_scaled = scaler.transform(x_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": xgb.XGBClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

"""Evaluation Model"""

def evaluate_model(model, x_data, y_data, data_type="Validation"):
    # Predictions
    y_pred = model.predict(x_data)

    # Metrics
    acc = accuracy_score(y_data, y_pred)
    cm = confusion_matrix(y_data, y_pred, labels=[0, 1])  # Ensure both classes are included
    cr = classification_report(y_data, y_pred, zero_division=0)  # Handle zero division

    # Print metrics
    print(f"\n{data_type} Results:")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    plt.figure(figsize=(6, 6))
    sb.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model.__class__.__name__} Confusion Matrix ({data_type})")
    plt.show()
    print("\nClassification Report:")
    print(cr)

    return {"Model": model.__class__.__name__, f"{data_type} Accuracy": acc}

"""Training Data"""

# Train and evaluate models
results = []
for name, model in models.items():
    print(f"\nTraining and Evaluating {name}...")

    # Step 1: Training
    model.fit(x_train_scaled, y_train_balanced)
    print(f"{name} trained successfully.")

    # Step 2: Validation
    validation_result = evaluate_model(model, x_validation_scaled, y_validation, data_type="Validation")

    # Step 3: Testing
    test_result = evaluate_model(model, x_test_scaled, y_test, data_type="Test")

    # Combine results
    combined_result = {
        "Model": name,
        "Validation Accuracy": validation_result["Validation Accuracy"],
        "Test Accuracy": test_result["Test Accuracy"]
    }
    results.append(combined_result)

"""Comparing Models"""

summary = pd.DataFrame(results)
print("\nModel Comparison:")
print(summary)

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
sb.barplot(x='Model', y='Test Accuracy', data=summary, palette='Blues_d')
plt.title("Test Accuracy Comparison")
plt.xticks(rotation=45)
plt.show()

# Best Model
best_model = summary.loc[summary['Test Accuracy'].idxmax()]
print(f"\nBest Model: {best_model['Model']} with Test Accuracy: {best_model['Test Accuracy'] * 100:.2f}%")

"""# Py Torch

Creating Dataset
"""

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train)
y_train_tensor = torch.tensor(y_train)
x_test_tensor = torch.tensor(x_test)
y_test_tensor = torch.tensor(y_test)

train_dataset = CustomDataset(x_train_tensor, y_train_tensor)
test_dataset = CustomDataset(x_test_tensor, y_test_tensor)

"""Dataloader"""

# DataLoader with larger batch size
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""
 Simplified Neural Network Model"""

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Single hidden layer with 128 neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)         # Output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))          # Activation function
        x = self.sigmoid(self.fc2(x))       # Sigmoid activation for binary classification
        return x.squeeze()

"""Optimizer and Loss function"""

input_dim = x_train.shape[1]
nn_model = SimpleNN(input_dim)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

"""Train and Test Loop"""

def train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, epochs=20):
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluation Phase
        model.eval()
        test_loss = 0.0
        all_preds, all_labels = [], []
        with torch.inference_mode():  # Disable gradient computation
            for inputs, labels in test_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Collect predictions and true labels
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

        # Print results
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

    return np.array(all_preds), np.array(all_labels)

# Train and Evaluate the Neural Network
print("\nTraining and Evaluating PyTorch Neural Network...")
nn_preds, nn_labels = train_and_evaluate(nn_model, criterion, optimizer, train_loader, test_loader, epochs=20)

"""# Visualization

Visualizing Pytorch
"""

# Visualize PyTorch Neural Network Results
print("\nNeural Network Results:")
plot_metrics(nn_labels, nn_preds, title="Neural Network")

"""# Conclusion

# Conclusion

**Scikit-learn Random Forest:**

Random Forest, being an ensemble method, often performs well on tabular data by combining multiple decision trees. Its strengths lie in handling non-linear relationships and reducing overfitting compared to individual decision trees.

**PyTorch Neural Network:**

Neural networks have the potential to learn complex patterns and relationships in data. Their performance depends heavily on factors such as architecture, hyperparameter tuning, and the amount of training data.

**Comparison and Explanation:**

In this specific scenario, you might observe that the Random Forest model achieves a higher accuracy or F1 score compared to the neural network. This could be attributed to the following reasons:

1. **Data Suitability:** Random Forest is often a good starting point for tabular data, while neural networks might require more data and careful tuning for optimal performance.
2. **Hyperparameter Optimization:** Neural networks have many hyperparameters that need careful tuning, and finding the best combination can be time-consuming. Random Forest generally requires less hyperparameter tuning.
3. **Model Complexity:** The neural network you've defined is relatively simple. A more complex architecture might yield better results, but it would also increase the risk of overfitting.

**Further Improvements:**

- **Hyperparameter Tuning:** Experiment with different hyperparameters for both models to potentially improve their performance.
- **Feature Engineering:** Explore creating new features or transforming existing ones to provide more informative input to the models.
- **Data Augmentation:** For the neural network, consider data augmentation techniques if applicable to increase the training data size.
- **Model Architecture:** For the neural network, experiment with different architectures, such as adding more layers or using different activation functions.

By carefully analyzing the results, understanding the strengths and weaknesses of each model, and iteratively improving the models and data, you can aim for the best possible performance on your prediction task.
"""