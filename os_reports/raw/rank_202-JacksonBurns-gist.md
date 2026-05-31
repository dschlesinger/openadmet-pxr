"""
fine tune the MIST foundation model on the OpenADMET PXR challenge.

Assumes you have a train.csv (containing the available training data) and test.csv (containing the blind challenges test SMILES).

adapted from: https://huggingface.co/mist-models/mist-1.8B-dh61satt#fine-tuning-for-property-prediction
in an environment with Python 3.12:

"torch>=2.0.0" "transformers>=4.30.0" "datasets>=2.0.0" "smirk>=0.1.0" "accelerate>=0.26.0" "rdkit>=2022.0.0" pandas tensorboard bitsandbytes
"""

import torch
import torch.nn as nn
from rdkit import Chem
from smirk import SmirkTokenizerFast
from datasets import load_dataset
import pandas as pd
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from pathlib import Path


def kekulize_smiles(smiles):
    """Convert SMILES to kekulized form."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    Chem.Kekulize(mol)
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


class RegressionModel(nn.Module):
    """Model with encoder + regression task head."""

    def __init__(self, encoder, hidden_size=768, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.task_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        # Get encoder outputs
        encoder_output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # Use first token
        pooled = encoder_output.last_hidden_state[:, 0, :]

        # Regression prediction
        logits = self.task_head(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits.squeeze(-1), labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize SMILES strings (kekulized)."""
    # MIST was pretrained on kekulized SMILES
    kekulized = [kekulize_smiles(s) for s in examples["SMILES"]]
    return tokenizer(
        kekulized,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def main():
    # 1. Load dataset from CSV
    dataset = load_dataset(
        "csv",
        data_files={"train": "train.csv", "test": "test.csv"},
    )

    # 2. Load pretrained encoder and tokenizer
    model_path = "mist-models/mist-1.8B-dh61satt"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    test_smiles = list(dataset["test"]["SMILES"])
    all_smiles = list(dataset["train"]["SMILES"]) + test_smiles
    actual_max_len = 0
    for smi in all_smiles:
        kekulized = kekulize_smiles(smi)
        tokens = tokenizer.encode(kekulized, add_special_tokens=True)
        if len(tokens) > actual_max_len:
            actual_max_len = len(tokens)
    print(f"Max tokenized length: {actual_max_len}")

    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    # 3. Create regression model with task head
    model = RegressionModel(
        encoder=encoder,
        hidden_size=encoder.config.hidden_size,
        dropout=0.1,
    )

    # 4. Tokenize dataset
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=actual_max_len),
        batched=True,
        desc="Tokenizing",
    )

    # Rename target column to labels (Trainer expects this)
    dataset = dataset.rename_column("pEC50", "labels")

    # Set format for PyTorch
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # 5. Setup training arguments
    outdir = Path("./mist_finetune")
    outdir.mkdir(exist_ok=True)
    training_args = TrainingArguments(
        output_dir=outdir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        report_to="tensorboard",
        torch_compile=False,
        # fit in 24gb VRAM, you bastard!
        fp16=True,
        optim="adamw_8bit",
    )

    # 6. Create data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # 7. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # 8. Train!
    trainer.train(resume_from_checkpoint="mist_finetune/checkpoint-9320")

    # 9. Save final model
    trainer.save_model(outdir)
    tokenizer.save_pretrained(outdir)

    # 10. Inference 
    inputs = tokenizer(
        [kekulize_smiles(s) for s in test_smiles],
        padding=True,
        truncation=True,
        max_length=actual_max_len,
        return_tensors="pt",
    )

    # Move to device and run inference
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs["logits"].squeeze(-1).cpu()

    test_df = pd.read_csv("test.csv")
    test_df["pEC50"] = predictions.numpy()
    test_df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
