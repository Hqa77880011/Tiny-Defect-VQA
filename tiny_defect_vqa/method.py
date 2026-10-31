"""Five-intent, 32-answer mask-guided MobileNet/Transformer VQA."""
import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder, region

ANSWER_OFFSETS = (0, 2, 15, 25, 29, 32)


class Model(nn.Module):
    """Implement the paper's multiscale attention and low-rank gated fusion."""

    def __init__(self, config, initialize_pretrained=False):
        super().__init__()
        self.config = dict(config)
        width = config.get("width", 128)
        rank = config.get("rank", 128)
        self.variant = config.get("variant", "full")
        self.encoder = Encoder(width, initialize_pretrained and config.get("pretrained", False))
        self.words = nn.Embedding(config.get("vocabulary", 8192), width, padding_idx=0)
        self.position = nn.Embedding(24, width)
        if self.variant == "mobilenet_gru":
            self.language = nn.GRU(width, width, batch_first=True)
        else:
            layer = nn.TransformerEncoderLayer(width, 4, 2*width, dropout=.1, batch_first=True)
            self.language = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        if self.variant == "mobilenet_gru":
            self.baseline_answer = nn.Linear(2*width, 32)
            self.register_buffer("answer_weight", torch.ones(32))
            return
        self.mask = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, groups=width), nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, 1, 1))
        self.visual = nn.Linear(width, width)
        self.query = nn.Linear(width, width)
        self.attention = nn.Linear(width, 1)
        self.alpha = nn.Parameter(torch.ones(()))
        self.left, self.right = nn.Linear(width, rank), nn.Linear(width, rank)
        self.fusion, self.visual_residual = nn.Linear(rank, width), nn.Linear(width, width)
        self.query_residual, self.gate = nn.Linear(width, width), nn.Linear(2*width, width)
        self.norm, self.answer = nn.LayerNorm(width), nn.Linear(width, 32)
        self.register_buffer("answer_weight", torch.ones(32))

    def forward(self, batch):
        v8, v16, v32 = self.encoder(batch["x"])
        token_ids = batch["question"].long()
        valid = token_ids != 0
        if token_ids.shape[1] > 24 or not valid.any(-1).all():
            raise ValueError("Questions require 1..24 tokens including nonpadding content")
        words = self.words(token_ids)+self.position(torch.arange(token_ids.shape[1], device=token_ids.device))[None]
        if self.variant == "mobilenet_gru":
            language, _ = self.language(words)
        else:
            language = self.language(words, src_key_padding_mask=~valid)
        question = (language*valid[..., None]).sum(1)/valid.sum(1, keepdim=True)
        tokens = v8.flatten(2).transpose(1, 2)
        if self.variant == "mobilenet_gru":
            logits = self.baseline_answer(torch.cat([v32.mean((2, 3)), question], -1))
            mask = torch.zeros_like(v8[:, :1])
            attention = torch.full_like(tokens[..., 0], 1/tokens.shape[1])
        else:
            mask = self.mask(v8).sigmoid()
            score = self.attention(torch.tanh(self.visual(tokens)+self.query(question)[:, None])).squeeze(-1)
            if self.variant != "no_mask":
                score = score+self.alpha*mask.flatten(1)
            attention = score.softmax(-1)
            visual = (attention[..., None]*tokens).sum(1)+v16.mean((2, 3))+v32.mean((2, 3))
            interaction = self.fusion(self.left(visual)*self.right(question))
            gate = self.gate(torch.cat([visual, question], -1)).sigmoid()
            if self.variant == "no_gate":
                gate = torch.ones_like(gate)
            fused = self.norm(gate*interaction+(1-gate)*self.visual_residual(visual)+self.query_residual(question))
            logits = self.answer(fused)
        intent = batch["intent"].long()
        if ((intent < 0) | (intent >= 5)).any():
            raise ValueError("Unregistered inspection intent")
        answer_ids = torch.arange(32, device=logits.device)[None]
        offsets = torch.tensor(ANSWER_OFFSETS, device=logits.device)
        allowed = (answer_ids >= offsets[intent, None]) & (answer_ids < offsets[intent+1, None])
        logits = logits.masked_fill(~allowed, -1e4)
        return {"logits": logits, "probability": logits.softmax(-1), "mask": mask, "attention": attention}

    def objective(self, batch, prediction):
        loss = F.cross_entropy(prediction["logits"], batch["y"].long(), weight=self.answer_weight)
        if self.variant not in {"no_localization", "mobilenet_gru"}:
            loss = loss+self.config.get("localization_weight", .5)*region(prediction["mask"], batch["mask"])
        return loss


def build_model(config, initialize_pretrained=False):
    """Construct the saved architecture, including contextual task adaptations."""
    if config.get("variant", "full") in {"mobivqa", "vqa_lite", "efficientad", "blip2", "anomalygpt"}:
        from .baselines import make
        return make(config, initialize_pretrained and config.get("pretrained", False))
    return Model(config, initialize_pretrained)
