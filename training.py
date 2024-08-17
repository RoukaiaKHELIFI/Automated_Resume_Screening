import pandas as pd
from datasets import DatasetDict, Dataset, load_metric
from sklearn.model_selection import train_test_split
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset

ds = load_dataset("ganchengguang/resume_seven_class")

# Define label mapping outside the function
label_mapping = {'PI': 0, 'Exp': 1, 'Sum': 2, 'Edu': 3, 'QC': 4, 'Skill': 5, 'Obj': 6, '': -1}

def data_preprocessing(ds):
    # Extracting text and labels from DatasetDict
    train_data = ds['train']

    # Split text and labels correctly
    text_labels = [text.split('\t') for text in train_data['text']]
    df = pd.DataFrame(text_labels, columns=['label', 'text'])

    # Map labels to integers
    df['label'] = df['label'].map(label_mapping)

    # Remove any rows with invalid labels
    df = df[df['label'] != -1]

    # Split into training and temporary sets (70% training, 30% temporary)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)

    # Split temporary set into validation and test sets (50% validation, 50% test)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Create Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    return train_dataset, val_dataset, test_dataset

# Load tokenizer and model
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=len(label_mapping) - 1)  # exclude invalid label

train_dataset, val_dataset, test_dataset = data_preprocessing(ds)

# Tokenize the data
def preprocess_function(examples):
    texts = examples['text']
    return tokenizer(texts, truncation=True, padding=True)

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# Create a new DatasetDict with tokenized datasets
tokenized_ds = DatasetDict({
    'train': tokenized_train,
    'validation': tokenized_val,
    'test': tokenized_test
})

# Define a data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Load the accuracy metric
metric = load_metric("accuracy")

# Define a function to compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions, references=torch.tensor(labels))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',  # Log directory for TensorBoard
    logging_steps=10,  # Log every 10 steps
    save_steps=500,  # Save checkpoint every 500 steps
    eval_strategy="steps",  # Evaluate during training at every logging step
    eval_steps=500,  # Evaluate every 500 steps
    load_best_model_at_end=True,  # Load the best model found at the end of training
    metric_for_best_model="accuracy",  # Use accuracy to select the best model
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./albert-finetuned')
tokenizer.save_pretrained('./albert-finetuned')

# Evaluate the model on the test set
trainer.evaluate(tokenized_ds['test'])
