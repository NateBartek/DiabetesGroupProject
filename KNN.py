# Diabetes Prediction(K-Nearest Neighbors Model)
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Function to process each dataset
def run_knn_model(file_name, target_column, dataset_name):

    print("Dataset:", dataset_name)
    print("-------------------------")

    #Load dataset
    df = pd.read_csv(file_name)

    df = df.sample(n=20000, random_state=42)


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

    train_acc = knn.score(X_train, y_train)
    test_acc = knn.score(X_test, y_test)

    y_prob = knn.predict_proba(X_test)

    print("Training Accuracy:", round(train_acc, 4))
    print("Testing Accuracy:", round(test_acc, 4))

    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    print("ROC-AUC Score:", round(roc_auc, 4))


    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.show()

    y_prob = knn.predict_proba(X_test)

    from sklearn.metrics import RocCurveDisplay

    # Only run for multi-class datasets
    if len(y.unique()) > 2:

        fig, ax = plt.subplots()

        class_names = ['Non-Diabetic', 'Pre-Diabetic', 'Diabetic']

        for i in range(len(class_names)):
            RocCurveDisplay.from_predictions(
                y_test == i,
                y_prob[:, i],
                name=class_names[i],
                ax=ax
            )

        plt.title(f"ROC Curve - {dataset_name}")
        plt.show()

    else:
        # Keep your original binary ROC here if you want
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        plt.plot(fpr, tpr)
        plt.title(f"ROC Curve - {dataset_name}")
        plt.show()


# Load dataset
data1 ="diabetes_012_health_indicators_BRFSS2015.csv"

# Run models on each dataset
run_knn_model(data1, "Diabetes_012", "Original Dataset")


