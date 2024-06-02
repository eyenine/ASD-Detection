import os
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc

# Load the trained model.
model = YOLO("yolov8n.yaml")

# Load the test dataset.
test_dataset = ["E:/UGC Project/ASD_Dataset1/test"]

# Check if the test dataset is accessible.
for image_path in test_dataset:
    if not os.access(image_path, os.R_OK):
        raise PermissionError(f"Permission denied: '{image_path}'")

# Perform inference on the test dataset.
results = model(test_dataset)

# Get the predicted labels and ground truth labels.
predicted_labels = []
ground_truth_labels = []
for i in range(len(results)):
    predicted_labels.append(results[i].pred[0].numpy())
    ground_truth_labels.append(results[i].y[0].numpy())

# Compute the performance metrics.
precision = precision_score(ground_truth_labels, predicted_labels, average="macro")
recall = recall_score(ground_truth_labels, predicted_labels, average="macro")
f1_score = f1_score(ground_truth_labels, predicted_labels, average="macro")
accuracy = accuracy_score(ground_truth_labels, predicted_labels)
confusion_matrix = confusion_matrix(ground_truth_labels, predicted_labels)

# Calculate the AUC and ROC curve.
fpr, tpr, thresholds = roc_curve(ground_truth_labels, predicted_labels)
auc = auc(fpr, tpr)

# Print the performance metrics.
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1_score)
print("Accuracy:", accuracy)
print("Confusion matrix:", confusion_matrix)
print("AUC score:", auc)