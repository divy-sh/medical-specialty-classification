# Install LIME if needed
!pip install lime

from lime.lime_text import LimeTextExplainer
import numpy as np

# Ensure you have access to the original transcription text for test samples
# Merge raw X_test with original dataset to access 'transcription' by index
X_test_raw = X_test.reset_index(drop=True)
y_test_raw = y_test.reset_index(drop=True)

# Set up LIME explainer
class_names = list(label_encoder.classes_)

# Define prediction function compatible with LIME
def predict_proba(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return model.predict(padded)

explainer = LimeTextExplainer(class_names=class_names)

# Choose a test example
i = 25  # Change index for other samples - 21
sample_text = X_test_raw[i]
true_label = y_test_raw[i]

print("Original Text:\n", sample_text)
print("True Label:", true_label)

# Generate explanation
exp = explainer.explain_instance(sample_text, predict_proba, num_features=10, top_labels=1)

# Show in notebook
exp.show_in_notebook(text=sample_text)
