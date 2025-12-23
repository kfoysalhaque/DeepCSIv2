import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Number of labels
num_labels = 18
# Total number of samples in the confusion matrix
total_samples = 10000

# Custom labels
labels = ['ee', 'd6', '28', '8b', 'c2', '65', 'b3', '2d', 'e6',
          'c6', '88', '6c', '0d', 'a9', '49', '12', '04', 'bb']

# Initialize confusion matrix with zeros
confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)

# Generate the correct predictions (diagonal values)
correct_predictions = int(total_samples * 0.9656)

# Labels where errors will be concentrated
error_labels = [5, 10, 12]  # Concentrate errors on these labels

# Distribute correct predictions randomly across the diagonal
for i in range(num_labels):
    if i in error_labels:
        # Assign slightly fewer correct predictions to error-prone labels
        confusion_matrix[i, i] = np.random.randint(
            int(correct_predictions // num_labels * 0.8),
            int(correct_predictions // num_labels * 0.9)
        )
    else:
        # Assign more correct predictions to less error-prone labels
        confusion_matrix[i, i] = np.random.randint(
            int(correct_predictions // num_labels * 1.05),
            int(correct_predictions // num_labels * 1.2)
        )

# Total remaining incorrect predictions
remaining_predictions = total_samples - np.trace(confusion_matrix)

# Ensure remaining_predictions is non-negative
if remaining_predictions < 0:
    remaining_predictions = 0

# Distribute incorrect predictions, concentrating on specific labels
for i in error_labels:
    incorrect_samples = max(remaining_predictions // len(error_labels), 0)
    if incorrect_samples > 0:
        incorrect_distributions = np.random.choice(
            [x for x in range(num_labels) if x != i],
            incorrect_samples,
            replace=True
        )
        for j in incorrect_distributions:
            confusion_matrix[i, j] += 1

# Adjust remaining errors randomly for non-error-prone labels (to balance matrix size)
for i in range(num_labels):
    if np.sum(confusion_matrix[i]) < total_samples // num_labels:
        remaining_count = (total_samples // num_labels) - np.sum(confusion_matrix[i])
        if remaining_count > 0:
            j = np.random.choice([x for x in range(num_labels) if x != i])
            confusion_matrix[i, j] += remaining_count

# Normalize the confusion matrix to represent percentages
normalized_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100

# Round the values to the nearest integer for display purposes
rounded_confusion_matrix = np.rint(normalized_confusion_matrix).astype(int)

# Plotting the confusion matrix with integer percentages and custom labels
plt.figure(figsize=(10, 8))
sns.heatmap(rounded_confusion_matrix, annot=True, fmt="d", cmap="Blues",
            cbar=True, xticklabels=labels, yticklabels=labels)  # Use custom labels

# Set font sizes for ticks, labels, and title
plt.xticks(fontsize=30, rotation=90)  # Adjust the font size and rotation for x-ticks
plt.yticks(fontsize=30, rotation=0)  # Adjust the font size for y-ticks
plt.xlabel('Predicted Labels', fontsize=30)  # Adjust the font size for x-axis label
plt.ylabel('True Labels', fontsize=30)  # Adjust the font size for y-axis label
#plt.title('Normalized Confusion Matrix (97% Accuracy, Focused Errors)', fontsize=16)  # Adjust the font size for title

# Show the plot
plt.tight_layout()
plt.show()

# Calculate and print the overall accuracy
accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
