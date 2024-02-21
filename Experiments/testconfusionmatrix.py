from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


true_labels = [1, 0, 1, 0,1,0]
predicted_labels = [1, 1, 0, 0,1,1]

# Compute the confusion matrix
# cm = confusion_matrix(true_labels, predicted_labels)
cm = [[3959.5, 224.5], [1067.0, 228.0]]
print(cm)

# Plot the confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()