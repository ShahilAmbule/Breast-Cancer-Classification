import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# File Paths
OUTPUT_PATH = "breast-cancer-wisconsin.csv"

# Headers
HEADERS = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
           "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "CancerType"]

def read_data(path):
    data = pd.read_csv(path)
    return data

def get_headers(dataset):
    return dataset.columns.values

def add_headers(dataset, headers):
    dataset.columns = headers
    return dataset

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header], train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def handle_missing_values(dataset, missing_values_header, missing_label):
    return dataset[dataset[missing_values_header] != missing_label]

def random_forest_classifier(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

def dataset_statistics(dataset):
    print(dataset.describe())

def plot_feature_importances(model, feature_headers):
    importances = model.feature_importances_
    indices = pd.Series(importances, index=feature_headers).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=indices, y=indices.index)
    plt.title("Feature Importances")
    plt.show()

def plot_confusion_matrix(test_y, predictions):
    cm = confusion_matrix(test_y, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    dataset = pd.read_csv(OUTPUT_PATH)
    dataset_statistics(dataset)

    dataset = handle_missing_values(dataset, "BareNuclei", '?')
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])

    print("Train_X Shape :: ", train_x.shape)
    print("Train_Y Shape :: ", train_y.shape)
    print("Test_X Shape :: ", test_x.shape)
    print("Test_Y Shape :: ", test_y.shape)

    trained_model = random_forest_classifier(train_x, train_y)
    print("Trained model ::", trained_model)
    predictions = trained_model.predict(test_x)

    for i in range(len(test_y)):
        print(f"Actual outcome :: {list(test_y)[i]} and Predicted outcome :: {predictions[i]}")

    print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print("Test Accuracy ::", accuracy_score(test_y, predictions))
    print("Confusion Matrix:\n", confusion_matrix(test_y, predictions))

    # Plot feature importances
    plot_feature_importances(trained_model, HEADERS[1:-1])

    # Plot confusion matrix
    plot_confusion_matrix(test_y, predictions)

if __name__ == "__main__":
    main()
