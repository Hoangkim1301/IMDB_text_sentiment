import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import pandas as pd
import evaluate
import re

#set number of train and test data sets
num = 100

# Load data from CSV
def load_csv_data(file_path, split):
    data = pd.read_csv(file_path)
    return data
# Function to clean the text data
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove special characters (Optional)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Keeps only alphanumeric characters and spaces
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space and strip leading/trailing spaces
    return text

# Load the datasets
train_data = load_csv_data("./imdb_train.csv", "train")
test_data = load_csv_data("./imdb_test.csv", "test")
train_data.dropna(inplace=True) #origonal data is modified
test_data.dropna(inplace=True)



# Select necessary columns
train_data = train_data[["text", "label"]].sample(n=num, random_state=42)
test_data = test_data[["text", "label"]].sample(n=num, random_state=42)

# Clean the text data
train_data['text'] = train_data['text'].apply(clean_text)
test_data['text'] = test_data['text'].apply(clean_text)

# Split train_data into train and validation sets (80% train, 20% validation)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42) 

# Ensure the splits are DataFrames
train_data = pd.DataFrame(train_data)
val_data = pd.DataFrame(val_data)
test_data = pd.DataFrame(test_data)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

# Combine datasets into a DatasetDict
dataset = DatasetDict({
    "train": train_dataset, 
    "validation": val_dataset, 
    "test": test_dataset
    })

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize function
def tokenize_function(examples):
    #print(examples["text"])
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True)
    # Add labels to the tokenized output
    tokens["labels"] = examples["label"]
    return tokens

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

#train, val, test

# Prepare datasets for PyTorch
train_dataset = tokenized_datasets["train"].shuffle(seed=42)
val_dataset = tokenized_datasets["validation"].shuffle(seed=42)
test_dataset = tokenized_datasets["test"].shuffle(seed=42)

# Initialize the BERT model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2, 
    #output_attentions=True
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Define metrics
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to numpy if it's not already
    logits = logits[0] if isinstance(logits, tuple) else logits
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on the test set
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test Accuracy:", test_results["eval_accuracy"])

# Save the model and tokenizer
model.save_pretrained("./imdb-model")
tokenizer.save_pretrained("./imdb-model")

# Predict on entire test dataset
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(-1)

# Match predicted labels with original text
for i, (text, true_label, pred_label) in enumerate(zip(
    test_data['text'], 
    test_data['label'], 
    predicted_labels
)):
    print(f"Text {i}:")
    print(f"Text: {text}")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {pred_label}")
    print("Correct" if true_label == pred_label else "Incorrect")
    print("-" * 50)
