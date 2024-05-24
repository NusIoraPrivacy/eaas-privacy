from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Tokenize the dataset using the BERT tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
print(train_dataset.column_names)
train_dataset = train_dataset.remove_columns('label')
# Train a BERT model for masked language modeling (MLM) on the IMDb dataset
model = BertForMaskedLM.from_pretrained(model_name)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    evaluation_strategy="no",
    num_train_epochs=4,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# Use the trained model to generate new movie reviews
prompt = "The movie was"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Move input_ids to GPU
input_ids = input_ids.to('cuda')

# Generate text using the trained model with beam search
output = model.generate(input_ids, max_length=256, num_return_sequences=5, temperature=0.7, num_beams=5, top_k=50)
for i, sequence in enumerate(output):
    decoded_sequence = tokenizer.decode(sequence, skip_special_tokens=True)
    print(f"Generated Text {i + 1}: {decoded_sequence}")

