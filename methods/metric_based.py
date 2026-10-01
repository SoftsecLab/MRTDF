import torch


def get_ll(text, base_model, base_tokenizer, DEVICE):
    with torch.no_grad():
        tokenized = base_tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()


def get_lls(texts, base_model, base_tokenizer, DEVICE):
    return [get_ll(_, base_model, base_tokenizer, DEVICE) for _ in texts]
