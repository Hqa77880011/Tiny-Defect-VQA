"""Image-disjoint QA generation, train-only WordPiece and joint geometry."""
import json
import hashlib
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from .method import ANSWER_OFFSETS

INTENTS = ["presence", "morphology", "location", "extent", "disposition"]
MORPHOLOGIES = ["crack", "scratch", "hole", "contamination", "deformation", "missing component", "misplacement", "color change", "surface irregularity", "broken part", "excess material", "unknown morphology"]
TEMPLATES = [
    ["Is a defect visible in this item?", "Does this item contain a visible defect?", "Can a defect be seen on the item?"],
    ["What defect morphology is visible?", "Which morphology describes the defect?", "What is the visible defect shape type?"],
    ["Which image region contains the defect?", "Where in the image is the defect?", "In which grid cell is the defect located?"],
    ["What is the visible defect extent?", "How large is the visible defect extent?", "Which visible area category describes the defect?"],
    ["What is the inspection disposition?", "Which inspection action applies to this item?", "How should this item be handled after inspection?"],
]


def answers(mask, morphology, anomalous, logical=False):
    """Typed answers from the mask centroid/area and declared disposition policy."""
    mask = np.asarray(mask) > 0
    ratio = mask.mean()
    if anomalous and not mask.any():
        raise ValueError("An anomalous source needs a nonempty trusted mask for grounded QA")
    extent = 0 if ratio == 0 else 1 if ratio <= .005 else 2 if ratio <= .02 else 3
    morphology_index = MORPHOLOGIES.index(morphology) if morphology in MORPHOLOGIES else 11
    location = 0
    if mask.any():
        yy, xx = np.where(mask)
        row = min(2, int(3*yy.mean()/mask.shape[0]))
        col = min(2, int(3*xx.mean()/mask.shape[1]))
        location = 1+3*row+col
    disposition = 0 if not anomalous else 2 if logical or morphology_index == 11 or extent == 3 else 1
    return [int(anomalous), 2+(1+morphology_index if anomalous else 0), 15+location, 25+extent, 29+disposition]


def prepare(manifest, output, seed=11):
    """Split source images before generating exactly five QA records per image."""
    source = Path(manifest)
    sources = json.loads(source.read_text(encoding="utf-8"))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = [str(row["dataset"])+":"+str(row["id"]) for row in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source image IDs")
    declared = ["split" in row for row in sources]
    if any(declared) and not all(declared):
        raise ValueError("Supply every source split or let preparation split all sources")
    if not any(declared):
        strata = {}
        for i, row in enumerate(sources):
            strata.setdefault((row["dataset"], row["category"], bool(row["anomalous"])), []).append(i)
        for indices in strata.values():
            order = rng.permutation(indices)
            n = len(order)
            # Floor allocation is explicit; exact published inventories require a frozen manifest.
            a, b, c = int(.7*n), int(.775*n), int(.85*n)
            for split, selected in zip(["train", "selection", "calibration", "test"], [order[:a], order[a:b], order[b:c], order[c:]]):
                for i in selected:
                    sources[i]["split"] = split
    records = []
    for row, identifier in zip(sources, ids):
        if row["split"] not in {"train", "selection", "calibration", "test"}:
            raise ValueError("Unknown split role")
        image = source.parent / row["image"]
        with Image.open(image) as loaded:
            shape = loaded.size
        mask_path = source.parent / row["mask"] if row.get("mask") else None
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0 if mask_path else np.zeros(shape[::-1], bool)
        if mask.shape != shape[::-1]:
            raise ValueError("Image and mask shapes differ")
        labels = answers(mask, row.get("morphology", "unknown morphology"), row["anomalous"], row.get("logical", False))
        for intent in range(5):
            template = int(rng.integers(2)) if row["split"] == "train" else 2
            record = dict(row, source_id=identifier, intent=intent, template=template, question=TEMPLATES[intent][template], y=labels[intent], generator_version=1)
            record["record_id"] = hashlib.sha256(f"{identifier}\0{intent}\0{template}".encode("utf-8")).hexdigest()[:20]
            record["image"] = os.path.relpath(image, out)
            record["mask"] = os.path.relpath(mask_path, out) if mask_path else None
            records.append(record)
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.train_from_iterator((record["question"] for record in records if record["split"] == "train"), trainers.WordPieceTrainer(vocab_size=8192, special_tokens=["[PAD]", "[UNK]"]))
    tokenizer.save(str(out/"tokenizer.json"))
    for record in records:
        record["tokens"] = tokenizer.encode(record["question"]).ids[:24]
    (out/"records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (out/"ontology.json").write_text(json.dumps({"intents": INTENTS, "morphologies": MORPHOLOGIES, "answer_offsets": ANSWER_OFFSETS, "seed": seed, "tokenizer_size": tokenizer.get_vocab_size()}, indent=2), encoding="utf-8")


class Questions(torch.utils.data.Dataset):
    """Lazy RGB/mask loader with synchronized geometry and regenerated labels."""

    def __init__(self, path, split, size=256, pretrained=False, augment=False, normalization=None):
        path = Path(path)
        self.root, self.size, self.pretrained, self.augment = path.parent, size, pretrained, augment
        self.normalization = normalization
        records = json.loads(path.read_text(encoding="utf-8"))
        mapping = {}
        for record in records:
            if record["source_id"] in mapping and mapping[record["source_id"]] != record["split"]:
                raise ValueError("A source image crosses partitions")
            mapping[record["source_id"]] = record["split"]
        self.records = [record for record in records if record["split"] == split]
        if not self.records:
            raise ValueError("Requested split contains no questions")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(self.root/row["image"]) as loaded:
            image = loaded.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        if row.get("mask"):
            with Image.open(self.root/row["mask"]) as loaded:
                mask = loaded.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
        else:
            mask = Image.new("L", (self.size, self.size))
        if self.augment:
            if torch.rand(()) < .5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            angle = float(torch.empty(()).uniform_(-5, 5))
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)
            image = ImageEnhance.Brightness(image).enhance(float(torch.empty(()).uniform_(.9, 1.1)))
            image = ImageEnhance.Contrast(image).enhance(float(torch.empty(()).uniform_(.9, 1.1)))
        target = np.asarray(mask).copy() > 0
        labels = answers(target, row.get("morphology", "unknown morphology"), row["anomalous"], row.get("logical", False))
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy().transpose(2, 0, 1)/255)
        if self.normalization == "clip":
            x = (x-torch.tensor([.48145466, .4578275, .40821073])[:, None, None])/torch.tensor([.26862954, .26130258, .27577711])[:, None, None]
        elif self.pretrained:
            x = (x-torch.tensor([.485, .456, .406])[:, None, None])/torch.tensor([.229, .224, .225])[:, None, None]
        question = torch.zeros(24, dtype=torch.long)
        question[:len(row["tokens"])] = torch.tensor(row["tokens"])
        return {"raw_question": row["question"], "x": x, "mask": torch.from_numpy(target[None].astype(np.float32)), "question": question, "y": torch.tensor(labels[row["intent"]]), "intent": torch.tensor(row["intent"]), "index": torch.tensor(index)}
