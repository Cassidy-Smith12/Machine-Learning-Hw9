import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

column_names = ['ID', 'Age', 'SpectaclePrescription', 'Astigmatism', 'TearProductionRate', 'Lenses']
data = pd.read_csv("lenses.csv", names=column_names)

data = data.drop(columns=['ID'])

X = data.drop(columns=['Lenses'])
y = data['Lenses']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

dtree = DecisionTreeClassifier(random_state=0)

dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

feature_importances = pd.Series(dtree.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nMost Important Features:")
print(feature_importances)

plt.figure(figsize=(12, 8))
plot_tree(dtree, feature_names=X.columns, class_names=['No Lenses', 'Soft Lenses', 'Hard Lenses'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

text_representation = export_text(dtree, feature_names=list(X.columns))
print("\nText Representation of the Decision Tree:")
print(text_representation)
