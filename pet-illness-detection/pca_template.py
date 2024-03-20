import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



if __name__ == '__main__':
    file_path = 'data.csv'
    data = pd.read_csv(file_path)

    symptoms_columns = ['symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5']
    data_encoded = pd.get_dummies(data, columns=symptoms_columns)

    data_encoded = data_encoded.drop('AnimalName', axis=1)
    data_encoded['Dangerous'] = data_encoded['Dangerous'].map({'Yes': 1, 'No': 0})
    data_cleaned = data_encoded.dropna()


    X = data_cleaned.drop('Dangerous', axis=1)
    y = data_cleaned['Dangerous']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)


    cm = confusion_matrix(y_test, predictions)


    # cm_percentage = cm / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix with Percentages')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()





