import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score,RocCurveDisplay, ConfusionMatrixDisplay


diabetes_health_indicators_data = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

feature_matrix = diabetes_health_indicators_data.drop('Diabetes_012', axis=1)
feature_label = diabetes_health_indicators_data['Diabetes_012']

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
print(classification_report(labels_test, predicted_diabetes,
      target_names=['Non-Diabetic', 'Pre-Diabetic', 'Diabetic']))

print("--- ROC-AUC ---")
print(f"ROC AUC: {roc_auc_score(labels_test, predicted_diabetes_probability, multi_class='ovr'):.4f}")

#ROC AUC curve
fig, ax = plt.subplots()
for i, class_name in enumerate(['Non-Diabetic', 'Pre-Diabetic', 'Diabetic']):
    RocCurveDisplay.from_predictions(labels_test == i, predicted_diabetes_probability[:, i],
                                     name=class_name, ax=ax)
plt.title('ROC Curve - Decision Tree')
plt.show()



ConfusionMatrixDisplay.from_predictions(labels_test, predicted_diabetes,
                                        display_labels=['Non-Diabetic', 'Pre-Diabetic', 'Diabetic'])
plt.title('Confusion Matrix - Decision Tree')
plt.show()