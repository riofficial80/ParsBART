import transformers
import torch
import os
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from datasets import Dataset
from time import time
from sklearn.model_selection import train_test_split
import pandas as pd

# PARAMETERS BART BASE
# ==============================================================================
VOCAB_SIZE = 52000
MAX_POSITION_EMBEDDINGS = 256
ENCODER_LAYERS = 6
ENCODER_FFN_DIM = 3072
ENCODER_ATTENTION_HEADS = 6
DECODER_LAYERS = 6
DECODER_FFN_DIM = 3072
DECODER_ATTENTION_HEADS = 6
D_MODEL = 768
DROPOUT = 0.1
# ==============================================================================

# Initialize a BART-Base model
tokenizer = BartTokenizer.from_pretrained("tokenizer_bart_it")

# Tiny version of BART
model = BartForConditionalGeneration(
    BartConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        encoder_layers=ENCODER_LAYERS,
        encoder_ffn_dim=ENCODER_FFN_DIM,
        encoder_attention_heads=ENCODER_ATTENTION_HEADS,
        decoder_layers=DECODER_LAYERS,
        decoder_ffn_dim=DECODER_FFN_DIM,
        decoder_attention_heads=DECODER_ATTENTION_HEADS,
        d_model=D_MODEL,
        dropout=DROPOUT,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        is_encoder_decoder=True,
        decoder_start_token_id=tokenizer.eos_token_id,
    )
)


start_range = 101
end_range = start_range + 5

train_dataset = pd.DataFrame()
for file_counter in tqdm(range(start_range,end_range)):
  if (file_counter <= 25):
    df = pd.read_csv(f"/content/drive/MyDrive/Colab Notebooks/Persian BART/cleaned_data/df{float(file_counter)}.csv")
    train_dataset = pd.concat([train_dataset,df])
  elif (file_counter <= 90):
    df = pd.read_csv(f"/content/drive/MyDrive/Colab Notebooks/Persian BART/cleaned_data_saberi/df{float(file_counter)}.csv")
    train_dataset = pd.concat([train_dataset,df])
  elif (file_counter <= 115):
    df = pd.read_csv(f"/content/drive/MyDrive/Colab Notebooks/Persian BART/cleaned_data_akbari/df{float(file_counter)}.csv")
    train_dataset = pd.concat([train_dataset,df])

# train_dataset, validation_dataset = train_test_split(total_data, test_size=0.00248, random_state=42)
validation_dataset = pd.read_csv(f"/content/drive/MyDrive/Colab Notebooks/Persian BART/cleaned_data_akbari/df115.0.csv")
validation_dataset = validation_dataset.head(1000)


# Convert DataFrames into Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_dataset)
validation_dataset = Dataset.from_pandas(validation_dataset)

# perturbation in string: document_rotation, sentence_permutation
# perturbation in token : token_infilling, token_masking, token_deletion
perturbations = [
    "document_rotation",
    "sentence_permutation",
    "token_infilling",
    "token_masking",
    "token_deletion",
]

perturbations_text_domain = [
    "document_rotation",
    "sentence_permutation",
]

perturbations_token_domain = [
    "token_infilling",
    "token_masking",
    "token_deletion",
]


def collate_fn(examples):
    """
    Collate function to be used in the dataloader.
    It applies the perturbations to the examples and returns the batch.
    TODO: improve efficiency
    :param examples: list of examples
    :return: batch ready to be fed to the model
    """
    original_texts = [example["text"] for example in examples]
    perturbed_texts = [example["perturbed_text"] for example in examples]
    perturbation_functions = [example["perturbation_function"] for example in examples]

    input_ids = None
    for perturbed_text, perturbation_function in zip(perturbed_texts, perturbation_functions):
        if perturbation_functions in perturbations_text_domain:
            perturbed_input_ids = tokenizer(
                    perturbed_text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_POSITION_EMBEDDINGS
                )["input_ids"][0]
        else:
            perturbed_input_ids = tokenizer(
                    perturbed_text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_POSITION_EMBEDDINGS, add_special_tokens=False
                )["input_ids"][0]

        if input_ids is None:
            input_ids = perturbed_input_ids.unsqueeze(0)
        else:
            input_ids = torch.cat((input_ids, perturbed_input_ids.unsqueeze(0)), dim=0)

    tokenized_examples = {}
    # update the tokenized examples with the perturbed input ids and convert to tensors
    tokenized_examples["input_ids"] = input_ids
    # update the attention mask
    tokenized_examples["attention_mask"] = [
        [1 if token_id != tokenizer.pad_token_id else 0 for token_id in input_ids]
        for input_ids in tokenized_examples["input_ids"]
    ]
    tokenized_examples["attention_mask"] = torch.tensor(tokenized_examples["attention_mask"])
    
    tokenized_examples["labels"] = tokenizer(
        original_texts, padding="max_length", truncation=True, max_length=MAX_POSITION_EMBEDDINGS, return_tensors="pt"
    )["input_ids"]

    return tokenized_examples


# total_steps (1 epoch, see it5) = 103_000_000 / 64 = 1_609_375 -- 1_700_000
# warmup_steps = 1_700_000 * 0.01 = 17_000

# Prepare training arguments
training_args = transformers.TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=True,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=1500,
    weight_decay=0.01,
    save_strategy="steps",
    evaluation_strategy="steps",
    max_steps=179688,
    logging_steps=400,
    eval_steps=7812,
    save_steps=7812,
    save_total_limit=5,#load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    fp16=True,#dataloader_num_workers=24,
    learning_rate=1e-4,
)

# Initialize the trainer

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=collate_fn
)

checkpoint_dir = '/content/drive/MyDrive/Colab Notebooks/Persian BART/bart-it/model/checkpoint-156240'


# make sure use the checkpoint dir to resume training from the checkpoint
trainer.train(resume_from_checkpoint=checkpoint_dir)
# trainer.train()

# Evaluate the model
print(trainer.evaluate(validation_dataset))

# Save the model
# trainer.save_model("./best_model")