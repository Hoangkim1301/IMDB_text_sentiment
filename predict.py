import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import pandas as pd
import evaluate
import re
#import matplotlib.pyplot as plt
#import seaborn as sns
#from bertviz import head_view

# Load saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./imdb-model")
tokenizer = AutoTokenizer.from_pretrained("./imdb-model")

# Function to clean the text data
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove special characters (Optional)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Keeps only alphanumeric characters and spaces
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space and strip leading/trailing spaces
    return text

# Load test data
test_data = pd.read_csv("./imdb_test.csv")
test_data.dropna(inplace=True)  # Drop missing values
test_data = test_data.sample(n=100, random_state=42)

# Clean the text data
test_data['text'] = test_data['text'].apply(clean_text)

test_dataset = Dataset.from_pandas(test_data)

# Tokenize function
def tokenize_function(examples):
    #print(examples["text"])
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True)
    # Add labels to the tokenized output
    tokens["labels"] = examples["label"]
    return tokens

# Tokenize datasets
tokenized_datasets = test_dataset.map(tokenize_function, batched=True)

# Define metrics
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Initialize Trainer
# Define a Trainer for evaluation
trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate the model on the test dataset
test_results = trainer.evaluate(eval_dataset=tokenized_datasets)
print("Test Accuracy:", test_results.get("eval_accuracy", "N/A"))

# Open a text file to write results
with open('model_evaluation.txt', 'w', encoding='utf-8') as f:
    # Write overall accuracy
    f.write(f"Test Accuracy: {test_results.get('eval_accuracy', 'N/A')}\n\n")
    
    # Predict and write detailed results
    predictions = trainer.predict(tokenized_datasets)
    predicted_labels = predictions.predictions.argmax(-1)
    
    # Write individual prediction details
    for i, (text, true_label, pred_label) in enumerate(zip(
        test_data['text'], 
        test_data['label'], 
        predicted_labels
    )):
        result = f"Text {i}:\n"
        result += f"Text: {text}\n"
        result += f"True Label: {true_label}\n"
        result += f"Predicted Label: {pred_label}\n"
        result += f"Correct: {'Yes' if true_label == pred_label else 'No'}\n"
        result += "-" * 50 + "\n"
        
        f.write(result)

print("Results saved to model_evaluation.txt")
'''
# Predict on entire test dataset
predictions = trainer.predict(tokenized_datasets)
predicted_labels = predictions.predictions.argmax(-1)

# Match predicted labels with original text
for i, (text, true_label, pred_label) in enumerate(zip(
    test_data['text'], 
    test_data['label'], 
    predicted_labels
)):
    print(f"Text {i}:")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {pred_label}")
    print("Correct" if true_label == pred_label else "Incorrect")
    print("-" * 50)
'''
