"""Executable task adaptations of the paper's contextual comparison methods.

Widths, pretrained-state paths and task-specific decoders are explicit configuration
choices; compact random initialization is solely a functional-check setting.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder, region
from .method import ANSWER_OFFSETS

NAMES = ("mobivqa", "vqa_lite", "efficientad", "blip2", "anomalygpt")


def typed(logits, intent):
    offsets = torch.tensor(ANSWER_OFFSETS, device=logits.device)
    ids = torch.arange(32, device=logits.device)[None]
    allowed = (ids >= offsets[intent, None]) & (ids < offsets[intent + 1, None])
    return logits.masked_fill(~allowed, -1e4)


class Text(nn.Module):
    def __init__(self, vocabulary, width, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width, padding_idx=0)
        self.position = nn.Embedding(24, width)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width, 4, 2 * width, batch_first=True, dropout=0.1
            ),
            layers,
            enable_nested_tensor=False,
        )

    def forward(self, ids):
        return self.encoder(
            self.embedding(ids)
            + self.position(torch.arange(ids.shape[1], device=ids.device))[None],
            src_key_padding_mask=ids == 0,
        )


class Cross(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, question, image):
        update, attention = self.attention(question, image, image, need_weights=True)
        value = self.norm(question + update)
        return value + self.ff(value), attention


class MobiVQA(nn.Module):
    """Grid features, attention-based early exits and actual token pruning."""

    def __init__(self, config, pretrained=False):
        super().__init__()
        self.config = dict(config, grounding_supported=False)
        width = config.get("width", 128)
        self.vision = Encoder(width, pretrained)
        self.text = Text(config.get("vocabulary", 8192), width)
        self.q_attention, self.v_attention = nn.Linear(width, 1), nn.Linear(width, 1)
        self.first_q, self.first_v = nn.Linear(width, width), nn.Linear(width, width)
        self.answerability = nn.Linear(width, 2)
        depth = config.get("cross_layers", 2)
        self.cross = nn.ModuleList([Cross(width) for _ in range(depth)])
        self.heads = nn.ModuleList([nn.Linear(2 * width, 32) for _ in range(depth)])
        self.exit = nn.Linear(width + 3, depth)
        self.keep_ratio = config.get("keep_ratio", 0.5)
        self.first_threshold = config.get("first_threshold", 0.5)
        self.exit_threshold = config.get("exit_threshold", 0.5)
        self.register_buffer("answer_weight", torch.ones(32))

    def forward(self, batch):
        visual = self.vision(batch["x"])[1].flatten(2).transpose(1, 2)
        question = self.text(batch["question"])
        valid = batch["question"] != 0
        q = (
            self.q_attention(question)
            .squeeze(-1)
            .masked_fill(~valid, -1e4)
            .softmax(-1)[..., None]
            * question
        ).sum(1)
        v = (self.v_attention(visual).softmax(1) * visual).sum(1)
        first = self.answerability(F.relu(self.first_q(q)) * F.relu(self.first_v(v)))
        logits = typed(self.heads[0](torch.cat([q, v], -1)), batch["intent"])
        review = first.softmax(-1)[:, 1] < self.first_threshold
        active = (
            torch.arange(len(q), device=q.device)
            if self.training
            else torch.nonzero(~review, as_tuple=False).flatten()
        )
        current_q, current_v, current_valid = (
            question[active],
            visual[active],
            valid[active],
        )
        all_logits, exit_logits = [], None
        layers = torch.zeros(len(q), dtype=torch.long, device=q.device)
        planned = torch.full(
            (len(active),), len(self.cross) - 1, dtype=torch.long, device=q.device
        )
        processed = torch.zeros(len(q), dtype=torch.long, device=q.device)
        for index, layer in enumerate(self.cross):
            if not len(active):
                break
            processed[active] += current_v.shape[1]
            current_q, attention = layer(current_q, current_v)
            qp = (current_q * current_valid[..., None]).sum(1) / current_valid.sum(
                1, keepdim=True
            )
            vp = current_v.mean(1)
            output = typed(
                self.heads[index](torch.cat([qp, vp], -1)), batch["intent"][active]
            )
            logits = logits.index_copy(0, active, output)
            layers[active] = index + 1
            if self.training:
                all_logits.append(output)
            salience = (attention * current_valid[..., None]).sum(
                1
            ) / current_valid.sum(1, keepdim=True)
            if index == 0:
                entropy = -(salience * salience.clamp_min(1e-9).log()).sum(
                    -1
                ) / math.log(max(salience.shape[1], 2))
                summary = torch.stack(
                    [
                        entropy,
                        salience.max(-1).values,
                        output.softmax(-1).max(-1).values,
                    ],
                    -1,
                )
                decision = self.exit(torch.cat([qp, summary], -1))
                if self.training:
                    exit_logits = decision
                else:
                    certainty, planned = decision.softmax(-1).max(-1)
                    planned = torch.where(
                        certainty >= self.exit_threshold,
                        planned,
                        torch.full_like(planned, len(self.cross) - 1),
                    )
            keep = max(1, int(math.ceil(current_v.shape[1] * self.keep_ratio)))
            selected = salience.topk(keep, dim=-1).indices
            current_v = current_v.gather(
                1, selected[..., None].expand(-1, -1, current_v.shape[-1])
            )
            if not self.training:
                continuing = planned > index
                active, current_q, current_v, current_valid, planned = (
                    active[continuing],
                    current_q[continuing],
                    current_v[continuing],
                    current_valid[continuing],
                    planned[continuing],
                )
        return {
            "logits": logits,
            "probability": logits.softmax(-1),
            "mask": batch["x"][:, :1, ::16, ::16] * 0,
            "first_logits": first,
            "layer_logits": all_logits,
            "exit_logits": exit_logits,
            "intrinsic_review": (
                review if not self.training else torch.zeros_like(review)
            ),
            "exit_layer": layers,
            "processed_visual_tokens": processed,
        }

    def objective(self, batch, prediction):
        heads = prediction["layer_logits"]
        labels = batch["y"]
        loss = sum(
            F.cross_entropy(logits, labels, weight=self.answer_weight)
            for logits in heads
        ) / len(heads)
        correct = torch.stack(
            [logits.detach().argmax(-1) == labels for logits in heads], -1
        )
        answerable = correct.any(-1).long()
        earliest = correct.float().argmax(-1)
        earliest = torch.where(
            answerable.bool(), earliest, torch.full_like(earliest, len(heads) - 1)
        )
        return (
            loss
            + 0.2 * F.cross_entropy(prediction["first_logits"], answerable)
            + 0.2 * F.cross_entropy(prediction["exit_logits"], earliest)
        )


class SoftMoE(nn.Module):
    """Soft token-to-slot routing and attention inside each expert's slots."""

    def __init__(self, width, experts=4, slots=4):
        super().__init__()
        self.experts, self.slots = experts, slots
        self.routing = nn.Parameter(torch.randn(width, experts * slots) / width**0.5)
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(width, 4, batch_first=True) for _ in range(experts)]
        )
        self.networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
                )
                for _ in range(experts)
            ]
        )

    def forward(self, tokens, valid):
        routing = tokens @ self.routing
        dispatch = routing.masked_fill(~valid[..., None], -1e4).softmax(1)
        combination = routing.softmax(-1)
        slots = torch.einsum("ntk,ntd->nkd", dispatch, tokens).reshape(
            len(tokens), self.experts, self.slots, -1
        )
        output = []
        for index, (attention, network) in enumerate(
            zip(self.attentions, self.networks)
        ):
            current = slots[:, index]
            update, _ = attention(current, current, current)
            output.append(current + network(current + update))
        experts = torch.stack(output, 1).flatten(1, 2)
        return torch.einsum("ntk,nkd->ntd", combination, experts), dispatch


class VQALite(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.config = dict(config, grounding_supported=False)
        width = config.get("width", 128)
        self.vision, self.text = Encoder(width, pretrained), Text(
            config.get("vocabulary", 8192), width
        )
        self.moe = SoftMoE(width, config.get("experts", 4), config.get("slots", 4))
        self.answer = nn.Linear(width, 32)
        self.register_buffer("answer_weight", torch.ones(32))

    def forward(self, batch):
        visual = self.vision(batch["x"])[1].flatten(2).transpose(1, 2)
        question = self.text(batch["question"])
        tokens = torch.cat([visual, question], 1)
        valid = torch.cat(
            [
                torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device),
                batch["question"] != 0,
            ],
            1,
        )
        fused, dispatch = self.moe(tokens, valid)
        pooled = (fused * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        logits = typed(self.answer(pooled), batch["intent"])
        return {
            "logits": logits,
            "probability": logits.softmax(-1),
            "mask": batch["x"][:, :1, ::16, ::16] * 0,
            "routing": dispatch,
        }

    def objective(self, batch, prediction):
        loss = F.cross_entropy(
            prediction["logits"], batch["y"], weight=self.answer_weight
        )
        if "teacher_logits" in batch:
            temperature = self.config.get("distillation_temperature", 2.0)
            loss = loss + self.config.get(
                "distillation_weight", 0.5
            ) * temperature**2 * F.kl_div(
                F.log_softmax(prediction["logits"] / temperature, -1),
                (batch["teacher_logits"] / temperature).softmax(-1),
                reduction="batchmean",
            )
        return loss


class PDN(nn.Sequential):
    def __init__(self, outputs, width=128):
        super().__init__(
            nn.Conv2d(3, width, 4),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(width, 2 * width, 4),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(2 * width, 2 * width, 3),
            nn.ReLU(),
            nn.Conv2d(2 * width, outputs, 4),
        )


class EfficientAD(nn.Module):
    """PDN student/teacher + global-bottleneck AE, followed by a typed decoder."""

    def __init__(self, config, pretrained=False):
        super().__init__()
        self.config = dict(config, grounding_supported=True)
        channels = config.get("feature_channels", 384)
        width = config.get("pdn_width", 128)
        self.teacher, self.student = PDN(channels, width), PDN(2 * channels, width)
        self.teacher.requires_grad_(False)
        self.autoencoder = nn.Sequential(
            nn.Conv2d(3, width // 2, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(width // 2, width, 4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, channels, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.register_buffer("teacher_mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("teacher_std", torch.ones(1, channels, 1, 1))
        self.register_buffer("quantiles", torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
        self.register_buffer("answer_weight", torch.ones(32))

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def maps(self, image):
        with torch.no_grad():
            target = (self.teacher(image) - self.teacher_mean) / self.teacher_std
        first, second = self.student(image).chunk(2, 1)
        latent = self.autoencoder(image)
        reconstruction = self.decoder(
            F.interpolate(
                latent, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        return target, first, second, reconstruction

    @torch.no_grad()
    def fit_reference(self, dataset, batch_size, device):
        normal = torch.utils.data.Subset(
            dataset,
            [i for i, row in enumerate(dataset.records) if not row["anomalous"]],
        )
        if not len(normal):
            raise ValueError("EfficientAD requires normal training reference images")
        values = [
            self.teacher(batch["x"].to(device))
            for batch in torch.utils.data.DataLoader(normal, batch_size=batch_size)
        ]
        total = sum(value.numel() // value.shape[1] for value in values)
        mean = sum(value.sum((0, 2, 3), keepdim=True) for value in values) / total
        variance = (
            sum(
                (value - mean).square().sum((0, 2, 3), keepdim=True) for value in values
            )
            / total
        )
        self.teacher_mean.copy_(mean)
        self.teacher_std.copy_(variance.sqrt().clamp_min(1e-5))

    @torch.no_grad()
    def calibrate_maps(self, dataset, batch_size, device):
        self.eval()
        normal = torch.utils.data.Subset(
            dataset,
            [i for i, row in enumerate(dataset.records) if not row["anomalous"]],
        )
        if not len(normal):
            raise ValueError(
                "Map normalization requires independent normal calibration images"
            )
        maps = [[], []]
        for batch in torch.utils.data.DataLoader(normal, batch_size=batch_size):
            target, first, second, ae = self.maps(batch["x"].to(device))
            maps[0].append((target - first).square().mean(1).flatten())
            maps[1].append((ae - second).square().mean(1).flatten())
        self.quantiles.copy_(
            torch.stack(
                [
                    torch.quantile(
                        torch.cat(values), torch.tensor([0.9, 0.995], device=device)
                    )
                    for values in maps
                ]
            )
        )

    def forward(self, batch):
        target, first, second, reconstruction = self.maps(batch["x"])
        st = (target - first).square().mean(1, keepdim=True)
        ae = (reconstruction - second).square().mean(1, keepdim=True)

        def normalize(value, index):
            low, high = self.quantiles[index]
            return (value - low) / (high - low).clamp_min(1e-6)

        mask = (0.5 * normalize(st, 0) + 0.5 * normalize(ae, 1)).clamp(0, 1)
        binary = mask[:, 0] >= 0.5
        area = binary.float().mean((1, 2))
        present = binary.flatten(1).any(-1)
        yy, xx = torch.meshgrid(
            torch.arange(mask.shape[-2], device=mask.device),
            torch.arange(mask.shape[-1], device=mask.device),
            indexing="ij",
        )
        mass = binary.sum((1, 2)).clamp_min(1)
        row = (3 * (binary * yy).sum((1, 2)) / mass / mask.shape[-2]).long().clamp(0, 2)
        col = (3 * (binary * xx).sum((1, 2)) / mass / mask.shape[-1]).long().clamp(0, 2)
        extent = torch.where(
            area == 0, 0, torch.where(area <= 0.005, 1, torch.where(area <= 0.02, 2, 3))
        )
        answers = torch.stack(
            [
                present.long(),
                torch.where(present, 14, 2),
                torch.where(present, 16 + 3 * row + col, 15),
                25 + extent,
                torch.where(present, 31, 29),
            ],
            -1,
        )
        selected = answers.gather(1, batch["intent"][:, None]).squeeze(1)
        confidence = (
            0.5 + 0.49 * (mask.flatten(1).max(-1).values - 0.5).abs() * 2
        ).detach()
        offsets = torch.tensor(ANSWER_OFFSETS, device=mask.device)
        size = offsets[batch["intent"] + 1] - offsets[batch["intent"]]
        probability = ((1 - confidence) / (size - 1))[:, None].expand(-1, 32).clone()
        probability.scatter_(1, selected[:, None], confidence[:, None])
        logits = typed(probability.clamp_min(1e-8).log(), batch["intent"])
        return {
            "logits": logits,
            "probability": logits.softmax(-1),
            "mask": mask,
            "target": target,
            "student": first,
            "student_ae": second,
            "reconstruction": reconstruction,
        }

    def objective(self, batch, prediction):
        error = (prediction["target"] - prediction["student"]).square()
        cutoff = torch.quantile(error.detach().flatten(), 0.999)
        hard = error[error >= cutoff].mean()
        ae = F.mse_loss(prediction["reconstruction"], prediction["target"])
        consistency = F.mse_loss(
            prediction["student_ae"], prediction["reconstruction"].detach()
        )
        penalty = 0.0
        if "penalty_x" in batch:
            penalty = (
                self.student(batch["penalty_x"])[:, : prediction["target"].shape[1]]
                .square()
                .mean()
            )
        return hard + ae + consistency + penalty


class LoRALinear(nn.Module):
    def __init__(self, width, rank=8):
        super().__init__()
        self.base = nn.Linear(width, width)
        self.base.requires_grad_(False)
        self.left = nn.Parameter(torch.randn(width, rank) * 0.01)
        self.right = nn.Parameter(torch.zeros(rank, width))
        self.rank = rank

    def forward(self, value):
        return self.base(value) + (value @ self.left @ self.right) / self.rank


class CausalBlock(nn.Module):
    def __init__(self, width, lora=False):
        super().__init__()
        self.q = LoRALinear(width) if lora else nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = LoRALinear(width) if lora else nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, x, valid):
        n, length, width = x.shape

        def heads(value):
            return value.reshape(n, length, 4, width // 4).transpose(1, 2)

        q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
        score = q @ k.transpose(-1, -2) / (width // 4) ** 0.5
        mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        score = score.masked_fill(mask, -1e4).masked_fill(
            ~valid[:, None, None, :], -1e4
        )
        update = (score.softmax(-1) @ v).transpose(1, 2).reshape(n, length, width)
        x = self.norm(x + self.out(update))
        return x + self.ff(x)


class Foundation(nn.Module):
    """Frozen ViT/causal-LM with Q-Former or anomaly prompts + LoRA task adaptation."""

    def __init__(self, config, pretrained=False):
        super().__init__()
        self.config = dict(
            config, grounding_supported=config["variant"] == "anomalygpt"
        )
        self.anomaly = config["variant"] == "anomalygpt"
        width = config.get("foundation_width", config.get("width", 128))
        self.patch = nn.Conv2d(3, width, 16, stride=16)
        self.visual_position = nn.Parameter(
            torch.randn(1, (config.get("size", 256) // 16) ** 2, width) * 0.02
        )
        self.vision = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width, 4, 2 * width, dropout=0.0, batch_first=True
            ),
            config.get("vision_layers", 2),
            enable_nested_tensor=False,
        )
        self.words = nn.Embedding(config.get("vocabulary", 8192), width, padding_idx=0)
        self.language_position = nn.Embedding(64, width)
        self.language = nn.ModuleList(
            [
                CausalBlock(width, lora=self.anomaly)
                for _ in range(config.get("language_layers", 2))
            ]
        )
        for module in [
            self.patch,
            self.vision,
            self.words,
            self.language_position,
            self.language,
        ]:
            module.requires_grad_(False)
        self.visual_position.requires_grad_(False)
        if self.anomaly:
            for name, parameter in self.language.named_parameters():
                if name.endswith(".left") or name.endswith(".right"):
                    parameter.requires_grad_(True)
            self.pixel = nn.Sequential(
                nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
            )
            self.prompts = nn.Parameter(
                torch.randn(1, config.get("prompt_tokens", 4), width) * 0.02
            )
            self.prompt_learner = nn.Sequential(
                nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)
            )
            self.register_buffer(
                "normal_memory", torch.zeros(config.get("memory", 512), width)
            )
            self.register_buffer("memory_count", torch.zeros((), dtype=torch.long))
        else:
            self.queries = nn.Parameter(
                torch.randn(1, config.get("query_tokens", 8), width) * 0.02
            )
            self.query_self = nn.MultiheadAttention(width, 4, batch_first=True)
            self.query_cross = nn.MultiheadAttention(width, 4, batch_first=True)
            self.query_ff = nn.Sequential(
                nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
            )
        self.projection = nn.Linear(width, width)
        self.answer = nn.Linear(width, 32)
        self.register_buffer("answer_weight", torch.ones(32))

    def visual(self, image):
        token = self.patch(image).flatten(2).transpose(1, 2)
        return self.vision(token + self.visual_position)

    @torch.no_grad()
    def fit_reference(self, dataset, batch_size, device):
        if not self.anomaly:
            return
        normal = torch.utils.data.Subset(
            dataset,
            [i for i, row in enumerate(dataset.records) if not row["anomalous"]],
        )
        if not len(normal):
            raise ValueError("AnomalyGPT adaptation needs normal training support")
        features = []
        for batch in torch.utils.data.DataLoader(normal, batch_size=batch_size):
            features.append(self.visual(batch["x"].to(device)).flatten(0, 1))
            if sum(len(value) for value in features) >= len(self.normal_memory):
                break
        values = torch.cat(features)[: len(self.normal_memory)]
        self.normal_memory[: len(values)].copy_(F.normalize(values, dim=-1))
        self.memory_count.fill_(len(values))

    def forward(self, batch):
        visual = self.visual(batch["x"])
        n, length, width = visual.shape
        if self.anomaly:
            distance = torch.zeros((n, length), device=visual.device)
            if self.memory_count:
                distance = (
                    1
                    - (
                        F.normalize(visual, dim=-1)
                        @ self.normal_memory[: int(self.memory_count)].T
                    )
                    .max(-1)
                    .values
                )
            pixel_logits = self.pixel(visual).squeeze(-1) + distance
            mask = pixel_logits.sigmoid().reshape(
                n, 1, int(length**0.5), int(length**0.5)
            )
            evidence = (visual * mask.flatten(2).transpose(1, 2)).mean(1)
            query = (
                self.prompts.expand(n, -1, -1) + self.prompt_learner(evidence)[:, None]
            )
        else:
            query = self.queries.expand(n, -1, -1)
            query = query + self.query_self(query, query, query)[0]
            query = query + self.query_cross(query, visual, visual)[0]
            query = query + self.query_ff(query)
            mask = batch["x"][:, :1, ::16, ::16] * 0
        prefix = self.projection(query)
        words = self.words(batch["question"])
        tokens = torch.cat([prefix, words], 1)
        valid = torch.cat(
            [
                torch.ones(prefix.shape[:2], dtype=torch.bool, device=prefix.device),
                batch["question"] != 0,
            ],
            1,
        )
        tokens = (
            tokens
            + self.language_position(
                torch.arange(tokens.shape[1], device=tokens.device)
            )[None]
        )
        for block in self.language:
            tokens = block(tokens, valid)
        position = prefix.shape[1] + (batch["question"] != 0).sum(-1) - 1
        pooled = tokens[torch.arange(n, device=tokens.device), position]
        logits = typed(self.answer(pooled), batch["intent"])
        return {"logits": logits, "probability": logits.softmax(-1), "mask": mask}

    def objective(self, batch, prediction):
        loss = F.cross_entropy(
            prediction["logits"], batch["y"], weight=self.answer_weight
        )
        if self.anomaly:
            loss = loss + self.config.get("localization_weight", 0.5) * region(
                prediction["mask"], batch["mask"]
            )
        return loss


class HFBlip2(nn.Module):
    """Actual Transformers BLIP-2 with local weights or a small verified config."""

    def __init__(self, config, pretrained=False):
        super().__init__()
        from transformers import (
            Blip2Config,
            Blip2ForConditionalGeneration,
            Blip2VisionConfig,
            Blip2QFormerConfig,
            OPTConfig,
            AutoTokenizer,
        )

        self.config = dict(config, grounding_supported=False)
        path = config.get("foundation_path")
        if "hf_config" in config:
            model_config = Blip2Config.from_dict(config["hf_config"])
            self.model = Blip2ForConditionalGeneration(model_config)
        elif path:
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                path, local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=True, use_fast=True
            )
            self.config["foundation_tokenizer_json"] = (
                tokenizer.backend_tokenizer.to_str()
            )
        else:
            width = config.get("foundation_width", 128)
            vision = Blip2VisionConfig(
                hidden_size=width,
                intermediate_size=2 * width,
                num_hidden_layers=config.get("vision_layers", 2),
                num_attention_heads=4,
                image_size=config.get("size", 256),
                patch_size=16,
            )
            query = Blip2QFormerConfig(
                hidden_size=width,
                intermediate_size=2 * width,
                num_hidden_layers=2,
                num_attention_heads=4,
                encoder_hidden_size=width,
                cross_attention_frequency=1,
            )
            text = OPTConfig(
                vocab_size=config.get("vocabulary", 8192),
                hidden_size=width,
                ffn_dim=2 * width,
                num_hidden_layers=config.get("language_layers", 2),
                num_attention_heads=4,
                max_position_embeddings=128,
                pad_token_id=0,
            )
            self.model = Blip2ForConditionalGeneration(
                Blip2Config(
                    vision_config=vision.to_dict(),
                    qformer_config=query.to_dict(),
                    text_config=text.to_dict(),
                    num_query_tokens=config.get("query_tokens", 8),
                )
            )
            # Nonzero random queries avoid degenerate zero-query normalization in
            # the small, unpretrained functional configuration.
            nn.init.normal_(self.model.query_tokens, std=0.02)
        self.config["hf_config"] = self.model.config.to_dict()
        self.model.vision_model.requires_grad_(False)
        self.model.language_model.requires_grad_(False)
        language_config = self.model.language_model.config
        width = getattr(language_config, "hidden_size", None) or language_config.d_model
        self.answer = nn.Linear(width, 32)
        self.register_buffer("answer_weight", torch.ones(32))
        self.tokenizer = None
        if "foundation_tokenizer_json" in self.config:
            from tokenizers import Tokenizer

            self.tokenizer = Tokenizer.from_str(
                self.config["foundation_tokenizer_json"]
            )

    def train(self, mode=True):
        super().train(mode)
        self.model.vision_model.eval()
        self.model.language_model.eval()
        return self

    def forward(self, batch):
        image = self.model.vision_model(
            pixel_values=batch["x"], interpolate_pos_encoding=True, return_dict=True
        ).last_hidden_state
        queries = self.model.query_tokens.expand(len(image), -1, -1)
        q = self.model.qformer(
            query_embeds=queries,
            encoder_hidden_states=image,
            encoder_attention_mask=torch.ones(
                image.shape[:2], dtype=torch.long, device=image.device
            ),
            return_dict=True,
        ).last_hidden_state
        prefix = self.model.language_projection(q)
        ids = batch["question"]
        valid = ids != 0
        if self.tokenizer is not None:
            if "raw_question" not in batch:
                raise ValueError(
                    "A pretrained foundation tokenizer requires raw_question strings"
                )
            encodings = self.tokenizer.encode_batch(batch["raw_question"])
            ids = torch.full(
                (len(encodings), 24),
                self.model.language_model.config.pad_token_id or 0,
                dtype=torch.long,
                device=image.device,
            )
            valid = torch.zeros_like(ids, dtype=torch.bool)
            for index, encoding in enumerate(encodings):
                tokens = encoding.ids[:24]
                ids[index, : len(tokens)] = torch.tensor(tokens, device=image.device)
                valid[index, : len(tokens)] = True
        embedding = self.model.language_model.get_input_embeddings()(ids)
        embeddings = torch.cat([prefix, embedding], 1)
        attention_mask = torch.cat(
            [
                torch.ones(prefix.shape[:2], dtype=torch.bool, device=image.device),
                valid,
            ],
            1,
        )
        if self.model.language_model.config.is_encoder_decoder:
            hidden = self.model.language_model.get_encoder()(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
        else:
            hidden = self.model.language_model.base_model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
        position = prefix.shape[1] + valid.sum(-1) - 1
        pooled = hidden[torch.arange(len(image), device=image.device), position]
        logits = typed(self.answer(pooled), batch["intent"])
        return {
            "logits": logits,
            "probability": logits.softmax(-1),
            "mask": batch["x"][:, :1, ::16, ::16] * 0,
        }

    def objective(self, batch, prediction):
        return F.cross_entropy(
            prediction["logits"], batch["y"], weight=self.answer_weight
        )


def make(config, pretrained=False):
    name = config["variant"]
    if name == "blip2" and config.get("foundation_backend") == "huggingface":
        return HFBlip2(config, pretrained)
    constructor = {
        "mobivqa": MobiVQA,
        "vqa_lite": VQALite,
        "efficientad": EfficientAD,
        "blip2": Foundation,
        "anomalygpt": Foundation,
    }[name]
    return constructor(config, pretrained)
