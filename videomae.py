#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import evaluate
import numpy as np

from data import GODData
from transformers import Trainer, TrainingArguments, VideoMAEConfig, VideoMAEForVideoClassification


# In[2]:


# load data
print("Loading data...")
train_dataset = GODData(
    subject="01", 
    session_id="01", 
    task="perception", 
    train=True, 
    limit_size=200,
)
eval_dataset = GODData(
    subject="01", 
    session_id="01", 
    task="perception", 
    train=False, 
    limit_size=50,
)

print(f"# train: {len(train_dataset):>5}\n# test: {len(eval_dataset):>5}")


# In[3]:


# instantiate model
print("Instantiating model...")
config = VideoMAEConfig(
    image_size=64,
    num_channels=3,
    num_frames=50,
    num_labels=150,
    problem_type="single_label_classification",
)

model = VideoMAEForVideoClassification(config)


# In[4]:


# data collation
def data_collator(datapoints):
    batch = {}
    batch["pixel_values"] = torch.stack([dp[0].permute(1, 0, 2, 3) for dp in datapoints])
    batch["labels"] = torch.stack([dp[1] for dp in datapoints])
    return batch


# In[5]:


# log metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# In[6]:


# instantiate trainer
print("Instantiating trainer...")
training_args = TrainingArguments(
    output_dir="test_trainer", 
    evaluation_strategy="epoch",
    num_train_epochs=500,
    per_device_train_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)


# In[7]:


print("Training...")
trainer.train()

