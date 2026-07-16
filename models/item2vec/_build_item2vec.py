"""Builds the 007-train-item2vec notebook.

Run:  uv run python models/item2vec/_build_item2vec.py
Throwaway builder — not part of the pipeline. Mirrors
`feature/engineer/_build_engineer.py`.
"""

from pathlib import Path

import nbformat as nbf


def build(path, cells_src):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells_src
    ]
    nb.cells = cells
    nbf.write(nb, path)
    print(f"wrote {path}  ({len(cells)} cells)")


# ============================================================ 007-train-item2vec
c = []
c.append(("md", """# 007 — Train Item2Vec (MovieLens, local single-machine)

Port of the reference `src/model_item2vec/main.py`, stripped of Ray Tune /
MLflow / distributed training. Pure PyTorch Lightning on one machine.

Drives `models/item2vec/train.py`, which runs two phases:

1. **Overfit sanity check** — train on one batch
   (`batch_sequences_overfit.jsonl`, produced by `006-prep-item2vec.ipynb`)
   for many epochs; expect `val_loss -> ~0`. Proves the model can learn.
2. **Full training** — train on `train_item_sequence.jsonl`, validate on
   `val_item_sequence.jsonl`, early-stop on `val_loss`, and save the best
   checkpoint + a copy of `idm.json` to `models/output/item2vec/final_model/`.

Inputs live under `feature/output/engineer/` (produced by the feature
engineering phase). Config: `configs/item2vec.yaml`."""))

c.append(("code", """import os
from pathlib import Path

from loguru import logger

# Project root = two levels up from this notebook (models/item2vec/).
ROOT = Path.cwd()
while ROOT.name and not (ROOT / "pyproject.toml").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
print("project root:", ROOT)
print("sequences present:", (ROOT / "feature/output/engineer/train_item_sequence.jsonl").exists())
print("overfit batch present:", (ROOT / "feature/output/engineer/batch_sequences_overfit.jsonl").exists())
print("idm present:", (ROOT / "feature/output/engineer/idm.json").exists())
"""))

c.append(("md", """## Run the training

`train.py` is a CLI entrypoint. Run it as a module from the project root so
the `models.item2vec` / `feature.id_mapper` package imports resolve:

```bash
uv run python -m models.item2vec.train --config configs/item2vec.yaml
```

The overfit sanity check runs first (drive `val_loss` to ~0 on one batch),
then the full training. Skip the sanity check with `--no-overfit`."""))

c.append(("code", """# Overfit sanity check + full training (single command).
!uv run python -m models.item2vec.train --config configs/item2vec.yaml
"""))

c.append(("md", """## Inspect the result

The final checkpoint + a copy of `idm.json` are written to
`models/output/item2vec/final_model/`. The embedding index space matches
`idm.item_to_index`, so item-id -> embedding lookup is a one-line
`model.embeddings(torch.tensor(idx))`."""))

c.append(("code", """import torch
from feature.id_mapper import IDMapper
from models.item2vec.model import SkipGram

final_dir = ROOT / "models/output/item2vec/final_model"
ckpt_path = final_dir / "best-checkpoint.ckpt"
print("checkpoint exists:", ckpt_path.exists())

if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    print("hparams keys:", list(hparams.keys()))

    # Rebuild the model from the checkpoint weights and look up one embedding.
    idm = IDMapper().load(str(final_dir / "idm.json"))
    model = SkipGram(num_items=len(idm.item_to_index), embedding_dim=hparams.get("embedding_dim", 64))
    print("vocab size:", len(idm.item_to_index))
    print("sample item_id -> idx:", list(idm.item_to_index.items())[:3])
"""))

build(str(Path(__file__).parent / "007-train-item2vec.ipynb"), c)