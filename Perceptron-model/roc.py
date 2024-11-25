import pandas as pd
import matplotlib.pyplot as plt

# Load your data
roc_data = pd.read_csv('rocOutput.csv')
roc_curves = roc_data.groupby('model')
# Plot the ROC curves
plt.figure(figsize=(10, 8))


for model_name, group in roc_curves:
    plt.plot(group['fpr'], group['tpr'], label=model_name)

plt.title('ROC Curves for Different Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
