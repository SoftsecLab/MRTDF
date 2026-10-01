import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from methods.utils_custom import get_clf_results
from methods.metric_based import get_lls, get_ll

def get_likelihood(logits, labels):
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()

def compute_discrepancy_scores(texts, model, tokenizer, device, use_reference=False):
    model.eval()
    scores = []
    for text in tqdm(texts, desc="Computing Fast DetectGPT scores"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
        labels = inputs.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = model(**inputs).logits[:, :-1]
            if use_reference:
                logits_ref = model(**inputs).logits[:, :-1]
            else:
                logits_ref = logits_score
            score = get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)
            scores.append(score)
    return scores

def run_fast_detectgpt_experiment(args, data, model, tokenizer, test_only=False, no_auc=False, ckpt_dir=None):
    real_texts = data['train']['text']
    real_labels = data['train']['label']
    test_texts = data['test']['text']
    test_labels = data['test']['label']

    print("Scoring real train texts...")
    real_scores = compute_discrepancy_scores(real_texts, model, tokenizer, args.DEVICE)
    print("Scoring test texts...")
    test_scores = compute_discrepancy_scores(test_texts, model, tokenizer, args.DEVICE)

    x_train = np.expand_dims(real_scores, axis=-1)
    y_train = real_labels
    x_test = np.expand_dims(test_scores, axis=-1)
    y_test = test_labels

    name = "fast_detectgpt"
    train_res, test_res = get_clf_results(x_train, y_train, x_test, y_test, name, test_only, no_auc, ckpt_dir)
    acc_train, precision_train, recall_train, f1_train, auc_train = train_res
    acc_test, precision_test, recall_test, f1_test, auc_test = test_res

    print(f"{name} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
    print(f"{name} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

    return {
        'name': name,
        'general': {
            'acc_train': acc_train,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_train': f1_train,
            'auc_train': auc_train,
            'acc_test': acc_test,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'f1_test': f1_test,
            'auc_test': auc_test,
        }
    }

def run_fast_detectgpt_experiments(args, data, base_model, base_tokenizer, test_only=False, no_auc=False, ckpt_dir=None):
    return run_fast_detectgpt_experiment(
        args=args,
        data=data,
        model=base_model,
        tokenizer=base_tokenizer,
        test_only=test_only,
        no_auc=no_auc,
        ckpt_dir=ckpt_dir
    )
