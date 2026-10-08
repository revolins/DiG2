# DiG²: Discrete-interpolative Graph Generation

Official code for *Discrete-interpolative Graph Generation*.

DiG² is a single discrete flow
model trained jointly on structural graphs (Planar, SBM, Tree) and molecules (QM9, ZINC). It
conditions each graph on its domain through a learned domain embedding and a structural
prototype, restricts every domain to its own node/edge vocabulary, and samples with a
closed-form posterior-bridge reverse process with classifier-free guidance. Interpolating the
domain conditioning between QM9 and ZINC generates molecules beyond either training
distribution, evaluated zero-shot against GuacaMol and MOSES.

- [Installation](#installation)
- [Data](#data)
- [Evaluating the released model](#evaluating-the-released-model)
- [Training](#training)
- [Anchored fine-tuning](#anchored-fine-tuning)
- [Discrete interpolation on molecules](#discrete-interpolation-on-molecules)
- [Adding your own dataset](#adding-your-own-dataset)
- [Repository layout](#repository-layout)

## Installation

Linux with a CUDA 12.6 on GPU. `graph-tool` (used to score SBM
graphs) is only available from conda-forge.

```bash
conda env create -f environment.yml
conda activate dig2
bash scripts/install.sh        # MOSES metrics package, this repo (editable), compiles ORCA
```

`scripts/install.sh` ends by importing `dig2`, `graph_tool`, `rdkit` and `moses`. To run the
test suite: `pytest -m "not gpu and not data and not ckpt"` (fast, no downloads) or `pytest`
(everything; needs the data and the checkpoints below).

## Data

All eight datasets are downloaded and preprocessed automatically the first time they are
used, into `data/` (set `dataset.datadir=/abs/path` to put them elsewhere).

| Dataset | Used as | Source |
|---|---|---|
| Planar, SBM, Tree | domains of the multi-domain model | [Bergmeister et al.](https://github.com/AndreasBergmeister/graph-generation) |
| QM9 (heavy atoms) | domain | MoleculeNet / PyG |
| ZINC250k | domain | [GruM](https://github.com/harryjo97/GruM) splits |
| GuacaMol, MOSES | zero-shot reference sets; standalone single-domain training | [GuacaMol](https://github.com/BenevolentAI/guacamol), [MOSES](https://github.com/molecularsets/moses) |

The multi-domain loader (`dig2/data/unified/`) maps every domain onto one shared semantic
vocabulary (node types: a generic structural node and atom type, edge
types: no edge, generic structural edge, single, double, triple bond) and caches the joint
dataset under `data/unified/cache__*`. The cache key includes `dataset.unified.schema_version`,
the first build takes 15-20 minutes, and every later run spends a few minutes loading the cache
into RAM. Preprocessing the full GuacaMol and MOSES sets for single-domain training is much
slower, RDKit parses each molecule on one core (about 150 molecules/s, so roughly two hours for
GuacaMol's 1.3M molecules). Processed files are reused.

## Evaluating the released model

```bash
python -m dig2.train +experiment=eval
```

This generates 40 graphs per structural domain and 10,000 molecules per molecular domain and
scores them (structural: validity / uniqueness / novelty and degree, clustering, orbit and
spectral MMDs; molecular: validity, uniqueness, novelty, FCD and the MOSES metrics). Results go
to `outputs/<date>/<time>-dig2_eval/test_epoch0_res.txt`, generated graphs to
`generated_samples_rank0.pkl` in the same directory. The release checkpoint carries its own
architecture config, so no other options are needed. It is applied over the defaults in
`dig2/configs/`; options you pass on the command line still take precedence (for example
`model.prior_mode=default`), and one that changes the architecture fails at the strict load
instead of running a different model.

**Evaluation protocol.** Each domain samples from its own seeded random stream
(`sample.rng_safe_per_domain`, `sample.rng_safe_per_batch`), so a domain's samples do not
depend on the other domains' budgets. Repeated draws use `sample.trajectory_seed_offset=k`
(same initial noise, independent reverse trajectories). The paper's numbers score 40 graphs per
structural domain and 10,000 molecules per molecular domain with the strict and
literature-convention checks of `dig2/tools/lit_protocol.py`:

```bash
python -m dig2.train +experiment=eval sample.trajectory_seed_offset=202 \
    evaluation.mixed.per_domain_samples_test="{planar: 40, sbm: 40, tree: 40, qm9: 10000, zinc: 10000}"
for d in planar sbm tree qm9 zinc; do
  python -m dig2.tools.lit_protocol run --run-dir outputs/<date>/<time>-dig2_eval --domain $d --workers 12
done
```

Each command writes `lit_<domain>.json` into the run directory. It reports strict validity
(SBM block-model fit, planarity, tree) and the literature variants, along with V.U.N. as a
mean ± std over blocks of 40 graphs. For molecules it reports plain, relaxed and strict
validity, uniqueness and novelty. `python -m dig2.tools.lit_protocol data --domain sbm --split test`
scores a real split with the same checks.

Any Lightning checkpoint from `dig2.train` can be evaluated the same way with
`general.eval_resume=<path>`; `dig2.tools.export_ckpt` converts it to the release format.

## Training

```bash
python -m dig2.train                        # the five-domain model of the paper
python -m dig2.train general.gpus=[1]       # pick the GPU
python -m dig2.train +experiment=smoke      # 2 training steps + a small validation pass
```

The model in the paper trains for 10,000 epochs of 250 node-budget batches (256 nodes, graphs
of mixed domains in every batch) on one 24 GB GPU. Validation every 100 epochs samples and
scores 40/40/40/128/256 graphs (planar/sbm/tree/qm9/zinc). Multi-GPU (DDP) training is
supported via `general.gpus=[0,1]`; the paper's runs used one GPU.

Other configurations (experiment overlays in `dig2/configs/experiment/`):

| Overlay | What it trains |
|---|---|
| `readout_3_6_9_12` | progressive readout at depths 3/6/9/12 |

The main options, all in `dig2/configs/dig2.yaml`:

- `general.guidance_weight_by_domain`: classifier-free guidance weight per domain.
- `general.structural_encoder.*`: the structural encoder and its per-domain EMA prototypes.
- `general.progressive_readout.*`: per-depth readout heads, gate and deep supervision.
- `train.domain_loss_weight`: per-domain loss weights.
- `sample.sample_steps`: number of reverse steps.

## Anchored fine-tuning

Anchored fine-tuning improves the target domains of a trained model while
protecting the others. Only the edge pathway is trainable: attention q/k/v/x_out and the
edge mixers, the edge FFNs, and the edge readout heads (228 tensors, 3.98M parameters). The
loss is the target domains' denoising loss plus
λ · KL(frozen starting model ‖ fine-tuned model) on the protected domains' predictions
(`dig2/procedures/anchor_ft.py`).

```bash
# fresh SBM / planar training, run from the repo root
python -m dig2.tools.build_pool --domain sbm
python -m dig2.tools.build_pool --domain planar
# 25,000 steps, lambda = 12, starting from the multi-domain model
python -m dig2.train +experiment=anchor_ft general.eval_resume=checkpoints/dig2_base.ckpt
```

## Discrete interpolation on molecules

The model is conditioned on a domain through two learned objects. This works with a row of the domain
embedding table and the domain's structural prototype. Interpolation replaces ZINC's row in
both with `(1 - α) · QM9 + α · ZINC`, conditions on ZINC, and allows the union of the two
domains' atom vocabularies (`dig2/tools/domain_interp_common.py`). α > 1 extrapolates past
ZINC.

- At α = 1 the outputs are bitwise identical to plain ZINC conditioning.
- At α = 0 the model receives exactly QM9's conditioning rows. The noise prior and guidance
  weight stay at ZINC's, so α only moves the conditioning and never the sampling problem.
- Run `--endpoint-check` to verify both statements on your checkpoint. It exits 0 on success.

Generated molecules are compared with GuacaMol or MOSES molecules of the same size band
(`--ref-source guacamol|moses`). Molecule size is controlled separately from α by one of three
arms:

| Arm | Molecule sizes |
|---|---|
| `target_prior` | drawn from the reference band's size histogram. Use this arm for chemical claims: size is held fixed across α |
| `size_forced` | fixed at the band midpoint |
| `free_running` | from the size prior blended by α |

```bash
python -m dig2.tools.domain_interp --ckpt checkpoints/dig2.ckpt --endpoint-check

python -m dig2.tools.domain_interp --ckpt checkpoints/dig2.ckpt \
    --arm target_prior --bands 15_19,20_27 --alphas 0.0,0.5,1.0,1.5 \
    --n-samples 2000 --seed 0 --crn --ref-source guacamol \
    --save-smiles-dir outputs/domain_interp/smiles --tag s0 \
    --out-json outputs/domain_interp/downstream_s0.json
```

`--crn` pairs the noise across α, so differences between cells come from α alone. The first run
parses the reference set into `outputs/domain_interp/refcache`; to do that ahead of time, run
`python -m dig2.tools.smiles_bands --sources guacamol,moses`. Use `--no-moses-metrics` for a quick
smoke run, which skips FCD, SNN, Scaf and Filters.

Property targeting turns α into a dial for QED or logP. It fits α → property on some seeds and
evaluates the calibrated α on held-out seeds:

```bash
for s in 0 1 2 3 4; do
  python -m dig2.tools.domain_interp --ckpt checkpoints/dig2.ckpt \
      --arm target_prior --bands 15_19 --alphas 0.0,0.5,1.0,1.5 \
      --n-samples 2000 --seed $s --crn --ref-source guacamol \
      --save-smiles-dir outputs/domain_interp/smiles --tag s$s \
      --out-json outputs/domain_interp/downstream_s$s.json
done
python -m dig2.tools.property_calibrate \
    --smiles-dir outputs/domain_interp/smiles \
    --interp-json 'outputs/domain_interp/downstream_s*.json' \
    --band 15_19 --props qed,logp --fit-seeds 3,4 --eval-seeds 0,1,2 \
    --out outputs/domain_interp/property/calibration.json
```

## Adding your own dataset

A new domain needs a loader, a vocabulary mapping, and config entries. `add_comm20.yaml` is a
complete worked example of the config side.

1. **Raw data.** A structural dataset in the SPECTRE/Bergmeister pickle format can reuse
   `SpectreGraphDataset` (`dig2/data/spectre.py`): add its download URL there. Otherwise write a
   PyG `InMemoryDataset` with train/val/test splits, following `dig2/data/qm9.py`.
2. **Register the domain** in `dig2/data/unified/dataset.py`: add its name to
   `UnifiedDataset.CANONICAL_DOMAINS` (and to `SPECTRE_DOMAINS` if it uses `SpectreGraphDataset`),
   add its type (`"synthetic"` or `"molecular"`) to the domain-type tables in `UnifiedDataset.__init__`
   and `dig2/data/unified/datamodule.py`, and add the loader call in `_instantiate_domain_datasets`.
3. **Vocabulary.** Map its local node and edge classes to the shared semantic vocabulary in
   `_SEMANTIC_NODE_MAPS_BASE` and `SEMANTIC_EDGE_MAPS` (a structural domain uses `[0]` and
   `[0, 1]`). These maps define the domain's vocabulary V_d, which the model enforces in the loss
   and at sampling. A new atom type needs a new entry in `SEMANTIC_NODE_VOCAB` **and** a bump of
   `dataset.unified.schema_version` (the vocabulary size is not part of the cache key).
4. **Metrics.** Structural domains are scored by `CombinedSpectreSamplingMetrics`
   (`dig2/metrics/spectre_utils.py`): add the name to its `SYNTHETIC_DOMAIN_NAMES` and, if the
   domain has a validity criterion (like `eval_acc_planar_graph`), register it in `domain_validity`.
   Molecular domains get an RDKit evaluator automatically (`dig2/pipeline.py`).
5. **Config.** Add the domain to `dataset.name`, `dataset.unified.domain_order`,
   `general.guidance_weight_by_domain`, `train.domain_loss_weight` and
   `evaluation.mixed.per_domain_samples[_test]`.
6. **Train from scratch.** The number of domains sets the shape of the domain embedding table,
   the prototype bank and the readout gate, so existing checkpoints do not load into a model
   with a different domain set.

## Repository layout

```
dig2/
  train.py                  entry point (Hydra); configs/ holds dig2.yaml + experiment overlays
  pipeline.py               datamodule, metrics and feature providers for a config
  checkpoint.py             strict loading, release-checkpoint format
  model/                    GraphDiscreteInterpolativeModel, graph transformer, progressive
                            readout, structural encoder, extra features
  interpolant/              noise (limit) distributions, time distortion, discrete sampling
  procedures/               anchored fine-tuning, structural training pools
  data/                     QM9, ZINC, GuacaMol, MOSES, SPECTRE graphs, the unified loader
  metrics/                  training losses, structural (MMD/ORCA/validity) and molecular metrics
  tools/                    checkpoint export, interpolation, property targeting,
                            literature-protocol scoring
tests/                      pytest suite
```
