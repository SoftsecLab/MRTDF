import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from transformers import DebertaV2Config
import torch.nn.functional as F
from transformers import DebertaV2Tokenizer, DebertaV2Model
from config import DEVICE, SBERT_PATH, DEBERTA_PATH, STYLE_CLASSIFIER_FULL_PATH, require_path


sbert_tokenizer = None
sbert_model = None


def get_sbert_semantic_embedding(text):
    global sbert_tokenizer, sbert_model
    if sbert_model is None:
        require_path(SBERT_PATH, 'SBERT_PATH')
        sbert_tokenizer = AutoTokenizer.from_pretrained(SBERT_PATH)
        sbert_model = AutoModel.from_pretrained(SBERT_PATH).to(DEVICE).eval()
    if not isinstance(text, str):
        text = ""
    inputs = sbert_tokenizer(text=text, return_tensors="pt", padding=True, truncation=True, max_length=384).to(DEVICE)
    with torch.no_grad():
        outputs = sbert_model(**inputs)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        embedding = (sum_embeddings / sum_mask).squeeze(0)
    return embedding.detach().cpu()


deberta_tokenizer = None


class StyleEncoderOnly(nn.Module):
    def __init__(self):
        super().__init__()
        require_path(DEBERTA_PATH, 'DEBERTA_PATH')
        self.backbone = DebertaV2Model.from_pretrained(DEBERTA_PATH)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        cls_layers = [layer[:, 0] for layer in outputs.hidden_states[-4:]]
        return torch.stack(cls_layers).mean(dim=0)

style_encoder_raw = None


@torch.no_grad()
def get_pure_style_embedding(text):
    require_path(DEBERTA_PATH, 'DEBERTA_PATH')
    global deberta_tokenizer, style_encoder_raw
    if deberta_tokenizer is None:
        deberta_tokenizer = DebertaV2Tokenizer.from_pretrained(DEBERTA_PATH)
    if style_encoder_raw is None:
        style_encoder_raw = StyleEncoderOnly().to(DEVICE).eval()
    inputs = deberta_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(DEVICE)
    inputs.pop("token_type_ids", None)
    style_vector = style_encoder_raw(**inputs)
    return style_vector.squeeze(0).cpu()


class StyleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        require_path(DEBERTA_PATH, 'DEBERTA_PATH')
        self.backbone = DebertaV2Model.from_pretrained(DEBERTA_PATH)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        cls_layers = [layer[:, 0] for layer in outputs.hidden_states[-4:]]
        cls_mean = torch.stack(cls_layers, dim=0).mean(dim=0)
        logits = self.classifier(cls_mean)
        return logits


style_classifier_eval = None


@torch.no_grad()
def get_classification_style_embedding(text):
    require_path(DEBERTA_PATH, 'DEBERTA_PATH')
    require_path(STYLE_CLASSIFIER_FULL_PATH, 'STYLE_CLASSIFIER_FULL_PATH')
    global deberta_tokenizer, style_classifier_eval
    if deberta_tokenizer is None:
        deberta_tokenizer = DebertaV2Tokenizer.from_pretrained(DEBERTA_PATH)
    if style_classifier_eval is None:
        style_classifier_eval = StyleClassifier().to(DEVICE)
        style_classifier_eval.load_state_dict(torch.load(STYLE_CLASSIFIER_FULL_PATH, map_location=DEVICE))
        style_classifier_eval.eval()
    inputs = deberta_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(DEVICE)
    inputs.pop("token_type_ids", None)
    logits = style_classifier_eval(**inputs)
    prob = F.softmax(logits, dim=-1)[0][1].item()
    pred = torch.argmax(logits, dim=-1).item()
    return prob, pred

