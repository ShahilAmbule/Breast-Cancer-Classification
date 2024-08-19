# Breast Cancer Classification using Random Forest

This project demonstrates the use of the Random Forest algorithm to classify breast cancer as benign or malignant based on various medical features.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Screenshot](#screenshot)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Breast cancer is a critical health issue that requires early detection and accurate diagnosis. This project uses the Random Forest algorithm to build a predictive model for breast cancer classification based on features like clump thickness, uniformity of cell size, and more.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin dataset. It contains various features related to breast cancer cell measurements and the diagnosis:

1. `CodeNumber`: An identifier for the sample.
2. `ClumpThickness`: Thickness of the clumps of cells.
3. `UniformityCellSize`: Uniformity of cell size.
4. `UniformityCellShape`: Uniformity of cell shape.
5. `MarginalAdhesion`: Marginal adhesion of cells.
6. `SingleEpithelialCellSize`: Size of single epithelial cells.
7. `BareNuclei`: Number of bare nuclei.
8. `BlandChromatin`: Chromatin texture.
9. `NormalNucleoli`: Number of normal nucleoli.
10. `Mitoses`: Number of mitoses.
11. `CancerType`: Type of cancer (target variable: Benign or Malignant).

## Libraries Used

The following libraries are used in this project:

- `pandas`: For data manipulation and analysis.
- `matplotlib`: For data visualization.
- `seaborn`: For advanced data visualization.
- `scikit-learn`: For machine learning algorithms and tools.

## Installation

To run this project, you need to have Python installed. Follow the steps below to set up the environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/breast-cancer-classification.git
    ```

2. Navigate to the project directory:
    ```bash
    cd breast-cancer-classification
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Requirements

The project requires the following Python libraries, which can be installed using the provided `requirements.txt` file:

- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Usage

1. Make sure you have the `breast-cancer-wisconsin.csv` file in the project directory. If not, you can download it from the UCI Machine Learning Repository or other sources and place it in the directory.

2. Run the script:
    ```bash
    python BreastCaner.py
    ```

3. The script will:
    - Load and preprocess the dataset.
    - Handle missing values and add headers.
    - Split the dataset into training and testing sets.
    - Train a Random Forest model.
    - Make predictions on the test set.
    - Display the actual and predicted outcomes.
    - Calculate and display the accuracy of the model.
    - Plot feature importances and the confusion matrix.

## Results

The script will output the accuracy of the model on the training and test sets, along with the confusion matrix and a plot of feature importances. It will also display a comparison of actual vs. predicted outcomes for a subset of the test data.
