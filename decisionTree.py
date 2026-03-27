import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay

datasets = [
    ("diabetes_012_health_indicators_BRFSS2015.csv", "Diabetes_012"),
    ("diabetes_binary_5050split_health_indicators_BRFSS2015.csv", "Diabetes_binary"),
    ("diabetes_binary_health_indicators_BRFSS2015.csv", "Diabetes_binary"),
]

for filename, target_col in datasets:
    print(f"\n--- {filename} ---")

    diabetes_health_indicators_data = pd.read_csv(filename)
    feature_matrix = diabetes_health_indicators_data.drop(target_col, axis=1)
    feature_label = diabetes_health_indicators_data[target_col]

    target_names = ['Non-Diabetic', 'Pre-Diabetic', 'Diabetic'] if feature_label.nunique() == 3 else ['Non-Diabetic', 'Diabetic']

    features_train, features_test, labels_train, labels_test = train_test_split(
        feature_matrix, feature_label, test_size=0.2, random_state=42)

    diabete_prediction_model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', max_depth=3)
    diabete_prediction_model.fit(features_train, labels_train)

    predicted_diabetes = diabete_prediction_model.predict(features_test)
    predicted_diabetes_probability = diabete_prediction_model.predict_proba(features_test)

    print("--- Accuracy ---")
    print(f"Training Accuracy : {accuracy_score(labels_train, diabete_prediction_model.predict(features_train)):.4f}")
    print(f"Test Accuracy     : {accuracy_score(labels_test, predicted_diabetes):.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(labels_test, predicted_diabetes, target_names=target_names))

    print("--- ROC-AUC ---")
    if feature_label.nunique() == 3:
        print(f"ROC AUC: {roc_auc_score(labels_test, predicted_diabetes_probability, multi_class='ovr'):.4f}")
    else:
        print(f"ROC AUC: {roc_auc_score(labels_test, predicted_diabetes_probability[:, 1]):.4f}")

    fig, ax = plt.subplots()
    for i, class_name in enumerate(target_names):
        RocCurveDisplay.from_predictions(labels_test == i, predicted_diabetes_probability[:, i],
                                         name=class_name, ax=ax)
    plt.title(f'ROC Curve - Decision Tree ({filename})')
    plt.show()

    ConfusionMatrixDisplay.from_predictions(labels_test, predicted_diabetes, display_labels=target_names)
    plt.title(f'Confusion Matrix - Decision Tree ({filename})')
    plt.show()