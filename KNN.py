# Diabetes Prediction(K-Nearest Neighbors Model)
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Function to process each dataset
def run_knn_model(file_name, target_column, dataset_name):

    print("Dataset:", dataset_name)
    print("-------------------------")

    #Load dataset
    df = pd.read_csv(file_name)

    #Check missing values
    df.isnull().sum()

    # If missing values exist, drop them
    df = df.dropna()

    #Check class balance
    df[target_column].value_counts()

    #Split features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    #Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.20, random_state=42)

    #HANDLING IMBALANCE
    # Combine X_train and y_train
    train_data = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)

    # Find majority and minority classes
    majority_class = train_data[train_data[target_column] == train_data[target_column].mode()[0]]
    minority_class = train_data[train_data[target_column] != train_data[target_column].mode()[0]]

    # Oversample minority class
    #used some AI to research how to implement this
    minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)

    # Combine again
    train_balanced = pd.concat([majority_class, minority_upsampled])

    # Split back
    X_train = train_balanced.drop(target_column, axis=1)
    y_train = train_balanced[target_column]

    #Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Train KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    #Predictions
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)

    #Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # ROC-AUC (handle binary vs multi-class)
    if len(y.unique()) > 2:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    else:
        roc_auc = roc_auc_score(y_test, y_prob[:, 1])

    #Print results
    print("Results:")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("-----------------------------------\n")

    # Return results for comparison
    return [dataset_name, accuracy, precision, recall, f1, roc_auc]


# Load all 3 datasets
data1 ="diabetes_012_health_indicators_BRFSS2015.csv"
data2 ="diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
data3="diabetes_binary_health_indicators_BRFSS2015.csv"


# Run models on each dataset
results = []
results.append(["Dataset", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC",])
results.append(run_knn_model(data1, "Diabetes_012", "Original Dataset"))
results.append(run_knn_model(data2, "Diabetes_binary", "Balanced Dataset"))
results.append(run_knn_model(data3, "Diabetes_binary", "Binary Dataset"))


# Compare Results
results_df = pd.DataFrame(results)
print("\nFinal Comparison Table")
print("-" *75 )
print(results_df)
