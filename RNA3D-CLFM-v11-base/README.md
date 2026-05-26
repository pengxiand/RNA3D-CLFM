# RNA3D-CLFM

RNA3D-CLFM (RNA 3D Contrastive Learning Foundation Model) is currently an RNA-only 3D project for two tasks:
- affinity prediction
- binding-site prediction/pretraining

Compatibility note:
- The repository folder and package names remain `BridgeBind3D` / `bridgebind3d` for now.
- This avoids import and path breakage while the project branding is upgraded to RNA3D-CLFM.

The design keeps a direct migration path to future protein-RNA models.

## Data strategy
Default workflow is RNAmigos2-only with one unified schema.

1. RNAmigos2 real pocket data
- `docking_data.csv` for affinity ranking
- `binary_data.csv` for affinity classification
- `json_pockets_expanded` for pocket-level structure

2. RNAmigos2 simulated data (as weak supervision source, not fixed 2.5D inductive bias)
- `pretrain_data/NR_chops/*.nx` for large-scale RNA 3D pretraining
- `ligand_db/*/(pdb|pdb_chembl|decoy_finder|robin)` for active/decoy contrastive pairs
- Use this data as labels/sampling source while upgrading encoder architecture to modern 3D graph models

3. Optional: GerNA-Bind
- kept only as an external comparison source (disabled by default)

## Data Construction Details (With Example)
This section explains exactly how raw files are converted into training manifests and
where augmented actives are used.

### 1) Raw data pieces and what they mean
RNAmigos2 side:
- `data/csvs/docking_data.csv`: pocket-ligand rows with continuous docking-style value
- `data/csvs/binary_data.csv`: pocket-ligand rows with binary native/non-native label
- `data/json_pockets_expanded/*.json`: pocket graph data used as RNA structure input
- `data/json_pockets_annotated/*.json`: same pocket graphs with extra ring annotations (`edge_annots`) for pretraining/analysis
- `data/ligand_db/<POCKET>/<mode>/(actives.txt, decoys.txt)`: active/decoy sets for contrastive/ranking construction

GerNA-Bind side (optional augmentation source):
- `data/Hariboss/hariboss.csv`: can be matched to RNAmigos2 pocket IDs (high-confidence augmentation source)
- `data/Robin/*.csv` and `data/Biosensor/*.csv`: mostly sequence-level supervision; many rows do not map to RNAmigos2 pocket IDs directly

### 2) JSON pocket composition
`json_pockets_expanded/*.json` follows graph-like JSON format:
- top-level: `nodes`, `links`, `directed`, `multigraph`
- each node typically has:
  - `id` (example: `1aju.A.21`)
  - `nt_code` (`A/C/G/U`)
  - `in_pocket` (boolean)
- each link typically contains source/target and relation type (for example `LW` in RNAmigos2 exports)

`json_pockets_annotated/*.json` extends this with:
- node field `edge_annots` (ring-style neighborhood annotation used by legacy/pretrain pipelines)

### 3) How manifests are built
`scripts/prepare_data_bridge.py` writes normalized CSVs under:
- `data/processed/manifests/`

Core outputs:
- `rnamigos_docking_manifest.csv`
- `rnamigos_binary_manifest.csv`
- `rnamigos_ligand_decoy_manifest.csv`
- `bridge_manifest.csv` (merged master table)

When active augmentation is enabled, script behavior is:
1. build RNAmigos2 decoy manifest from `ligand_db/*/*/(actives.txt, decoys.txt)`
2. build matched positive rows from selected GerNA sources
3. append positives to decoy manifest and drop duplicates by `(pocket_id, ligand_smiles, label_value)`
4. optionally export unmatched rows for transparency
5. optionally materialize merged `actives.txt` into a separate output tree

Important:
- Upstream RNAmigos2 files are not modified.
- Augmented ligand_db files are written to `data/processed/augmented_ligand_db/`.

### 4) Concrete example: pocket `2ESI_A_KAN_50`
After running augmentation:
- original active file: `data/data/ligand_db/2ESI_A_KAN_50/pdb_chembl/actives.txt` (1 line)
- augmented active file: `data/processed/augmented_ligand_db/2ESI_A_KAN_50/pdb_chembl/actives.txt` (2 lines)

The newly added active is also present in training CSV:
- `data/processed/manifests/rnamigos_ligand_decoy_manifest.csv`
  with source `Hariboss` and `label_value=1` for pocket `2ESI_A_KAN_50`.

### 5) Why one file still shows 1 active
If you inspect `data/data/ligand_db/.../actives.txt`, this is expected to stay unchanged.
Use one of these as training truth instead:
- `data/processed/manifests/rnamigos_ligand_decoy_manifest.csv` (preferred for trainer)
- `data/processed/augmented_ligand_db/.../actives.txt` (if downstream code reads text files directly)

### 6) Recommended command patterns
RNAmigos2-only manifest build:
```bash
python scripts/prepare_data_bridge.py \
  --rnamigos-root ../rnamigos2 \
  --build-ligand-decoy-manifest
```

Multi-source active augmentation (Hariboss/Robin/Biosensor):
```bash
python scripts/prepare_data_bridge.py \
  --rnamigos-root ../rnamigos2 \
  --gerna-root ../GerNA-Bind \
  --build-ligand-decoy-manifest \
  --augment-gerna-actives \
  --gerna-active-sources hariboss,robin,biosensor \
  --export-unmatched-gerna-actives \
  --materialize-augmented-ligand-db \
  --augmented-ligand-db-mode pdb_chembl \
  --augmented-ligand-db-out data/processed/augmented_ligand_db
```

## Representation Design (Pocket + Ligand)
This project now targets a modern two-tower design:
- one RNA 3D tower (pocket encoder)
- one ligand graph tower
- one shared embedding space for contrastive/ranking objectives
- one residue-level site head attached to the RNA tower

### Pocket 3D representation (recommended default)
Use RNAmigos2 pocket preprocessing only as data interface, while replacing legacy 2.5D-style encoders with 3D geometric encoders.

Input source:
- `json_pockets_expanded/*.json`

Node-level features:
- nucleotide identity (`A/C/G/U/N`)
- `in_pocket` flag
- local topological statistics (for example degree)
- 3D coordinates and local frame information (if available)

Edge-level features:
- relation / interaction type from RNAmigos2 graph construction
- distance buckets and optional directional geometry features

Recommended RNA pocket encoder candidates:
1. GVP-GNN (recommended first)
- Naturally models scalar/vector channels for 3D structure.
- Good balance between geometry fidelity and engineering complexity.

2. EGNN (strong alternative)
- Coordinate-aware message passing with simpler implementation.
- Competitive for 3D graph tasks with moderate compute.

3. SE(3)-aware models (advanced option)
- Higher geometric expressiveness, higher compute/engineering cost.
- Suggested for final ablation or best-performance run.

Why this default direction:
- keeps compatibility with current manifests
- removes hard dependency on legacy 2.5D assumptions
- better inductive bias for pocket-level 3D interaction learning

### Ligand representation (recommended default)
Use molecular graph as primary representation instead of sequence-only SMILES encoder.

Recommended ligand encoder candidates:
1. D-MPNN / MPNN (recommended first)
- Strong practical baseline for atom-bond message passing.
- Stable for ranking and binary active/decoy tasks.

2. AttentiveFP or GIN/GATv2 variants
- Useful alternatives for architecture ablation.

3. Graph transformer family (advanced option)
- Potentially stronger on large ligand diversity.
- Higher memory and tuning cost.

Fallback policy:
- Keep deterministic placeholder vectors only for smoke tests.

### Two-tower interaction and objectives (new default)
Two-tower setup:
- RNA tower: GVP-GNN or EGNN over pocket 3D graph
- Ligand tower: molecular graph encoder
- Projection heads: `z_rna`, `z_lig` in one metric space

Core objectives:
1. Contrastive loss (InfoNCE / multi-negative)
- Pull active pairs together, push decoys away.

2. Ranking loss (margin/listwise)
- Enforce active > decoy ordering within pocket.

3. Decoy-separation score loss
- Binary BCE or pairwise logistic on active/decoy labels.

4. Optional affinity regression head
- Use docking score/value when available.

5. Binding-site residue head
- Keep site BCE on RNA node embeddings.

### Practical ablation axes
1. RNA encoder: GVP vs EGNN vs relation-aware non-3D GNN.
2. Ligand encoder: D-MPNN vs AttentiveFP vs graph transformer.
3. Loss stack: contrastive only vs contrastive + ranking vs full multitask.
4. Decoy hardness: easy/medium/hard and pocket-aware mixing.

### Recommended first submission track
1. Pocket: RNAmigos2 JSON + coordinates, RNA tower uses GVP-GNN.
2. Ligand: RDKit atom-bond graph, ligand tower uses D-MPNN.
3. Interaction: two-tower projection + optional lightweight cross-attention fusion.
4. Losses: InfoNCE + margin ranking + active/decoy BCE + site BCE.
5. Report decoy regimes at 20/50/100 with strict scaffold+family split.

For internal Chinese planning notes and architecture figure, see:
- `docs_rna3d_encoder_plan_zh.md`
- `docs_downstream_execution_plan_zh.md`
- `docs_downstream_tasks_zh.md`

## Repository layout
- `configs/`: data source config and run settings
- `scripts/`: one-shot data preparation scripts
- `data/raw/`: symlink/copy views of upstream repos
- `data/processed/manifests/`: normalized CSV manifests
- `src/bridgebind3d/`: python package for loaders and tasks

## Quick start
1. Install minimal dependencies
```bash
pip install -r requirements.txt
```

2. Build RNAmigos2 manifests (including simulated data)
```bash
python scripts/prepare_data_bridge.py \
  --rnamigos-root ../rnamigos2 \
  --link-mode symlink \
  --build-pocket-node-manifest \
  --build-simulated-manifest \
  --build-ligand-decoy-manifest
```

Optional: include GerNA-Bind rows as additional supervision.
```bash
python scripts/prepare_data_bridge.py \
  --rnamigos-root ../rnamigos2 \
  --include-gerna \
  --gerna-root ../GerNA-Bind \
  --link-mode symlink \
  --build-pocket-node-manifest \
  --build-simulated-manifest \
  --build-ligand-decoy-manifest
```

Optional: augment RNAmigos2 ligand-decoy manifest with Hariboss positive actives
(keep RNAmigos2 decoys unchanged).
```bash
python scripts/prepare_data_bridge.py \
  --rnamigos-root ../rnamigos2 \
  --gerna-root ../GerNA-Bind \
  --link-mode symlink \
  --build-pocket-node-manifest \
  --build-simulated-manifest \
  --build-ligand-decoy-manifest \
  --augment-hariboss-actives
```

Generated extra file in this mode:
- `data/processed/manifests/hariboss_active_manifest.csv`

Optional: materialize ligand_db-style files with merged actives.txt
(recommended when downstream code directly reads `ligand_db/*/pdb_chembl/actives.txt`).
This writes to a separate output tree and does not modify upstream `rnamigos2`.
```bash
python scripts/prepare_data_bridge.py \
  --rnamigos-root ../rnamigos2 \
  --gerna-root ../GerNA-Bind \
  --build-ligand-decoy-manifest \
  --augment-hariboss-actives \
  --materialize-augmented-ligand-db \
  --augmented-ligand-db-mode pdb_chembl \
  --augmented-ligand-db-out data/processed/augmented_ligand_db
```

Then use:
- `data/processed/augmented_ligand_db/<POCKET>/pdb_chembl/actives.txt`
- `data/processed/augmented_ligand_db/<POCKET>/pdb_chembl/decoys.txt`

3. Check outputs
- `data/processed/manifests/bridge_manifest.csv`
- `data/processed/manifests/summary.json`

## How to run now (RNAmigos2-only)
BridgeBind3D currently provides data preparation and manifest unification.
It now also includes a unified multitask training scaffold with one shared interaction backbone and three heads.

Step 1: prepare manifests
- bash scripts/bootstrap_data.sh

Step 2: inspect generated files
- data/processed/manifests/rnamigos_simulated_pretrain_manifest.csv
- data/processed/manifests/rnamigos_ligand_decoy_manifest.csv
- data/processed/manifests/rnamigos_docking_manifest.csv
- data/processed/manifests/rnamigos_binary_manifest.csv
- data/processed/manifests/rnamigos_pocket_nodes_manifest.csv

Step 3: launch pretraining/fine-tuning using those manifests in your trainer
- Stage A pretrain: use simulated + decoy manifests
- Stage B affinity: use docking + binary manifests
- Stage C binding-site: use pocket-node manifest

Step 4: run unified multitask scaffold
- python scripts/train_unified_multitask.py --config configs/unified_multitask.yaml

Step 5: run dual-encoder downstream profile (recommended internal profile)
- python scripts/train_unified_multitask.py --config configs/dual_encoder_gvp_dmpnn.yaml
- Legacy config alias still available: `configs/two_tower_gvp_dmpnn.yaml`.
- This profile defines RNA tower = GVP-GNN and ligand tower = D-MPNN as target architecture settings.

Step 6: run EGNN-based dual-encoder profile (recommended if you prioritize custom RNA 3D geometric modeling)
- python scripts/train_unified_multitask.py --config configs/dual_encoder_egnn_dmpnn.yaml
- Legacy config alias still available: `configs/two_tower_egnn_dmpnn.yaml`.
- This profile sets RNA tower to an EGNN-style encoder and keeps ligand tower in D-MPNN-style settings.
- Current scaffold supports a staged objective stack: contrastive + ranking + docking/affinity-like score + decoy BCE + ligand-conditioned site head.

Step 7: run graph-native EGNN+GIN dual-encoder trainer (new)
- python scripts/train_dual_encoder.py
- or: python scripts/train_two_tower.py --config configs/egnn_gin_dual_encoder.yaml
- This trainer uses explicit graph batches: RNA tower = EGNN, ligand tower = GINE/GIN, and three heads = screening/affinity/site.

Important note:
- The scaffold supports two featurizer modes via `configs/unified_multitask.yaml`:
  - `featurizer.mode: real` (default): pocket JSON-driven RNA residue features + optional RDKit atom features.
  - `featurizer.mode: placeholder`: deterministic random tokens for smoke testing.
- If RDKit is unavailable, ligand featurization falls back to deterministic placeholder vectors.

## Pretrain I/O definition
Use this as the contract for your trainer.

Two-tower output contract (shared across tasks):
- `h_rna`: residue/token-level RNA embeddings from RNA 3D tower
- `h_lig`: atom/token-level ligand embeddings from ligand graph tower
- `z_rna`: pooled/projection RNA embedding
- `z_lig`: pooled/projection ligand embedding
- `z_pair`: optional fused pair embedding (if fusion head is enabled)

Task heads in current scaffold:
- `rank_head(z_rna, z_lig or z_pair) -> rank_score`
- `decoy_head(z_rna, z_lig or z_pair) -> decoy_logit`
- `dock_head(z_rna, z_lig or z_pair) -> dock_score`
- `site_head(h_rna) -> site_logits`

Contrastive/ranking objectives in downstream profile:
- In-batch multi-negative InfoNCE in shared embedding space (`z_rna`, `z_lig`).
- For each positive (active) pair, sample pocket-aware decoys and optimize active > decoy ranking.
- Add decoy separation head (BCE or pairwise logistic) to explicitly push decoys away.

### A. Simulated graph pretrain
Input table: rnamigos_simulated_pretrain_manifest.csv

Required input fields per row:
- task = binding_site_self_supervised
- target_id (graph id)
- target_structure_path (path to .nx graph)
- extra.annotated_graph_path (optional annotation path)

Model input tensors:
- RNA 3D graph features from target_structure_path

Model output:
- graph-level embedding or node-level embedding (depending on pretext objective)

Supervision:
- self-supervised objective (no explicit label column)

### B. Active/decoy interaction pretrain
Input table: rnamigos_ligand_decoy_manifest.csv

Required input fields per row:
- task = affinity_classification
- pocket_id / target_structure_path (RNA pocket graph)
- ligand_smiles
- label_value in {0, 1}

Model input tensors:
- RNA pocket 3D graph
- ligand molecular graph

Model output:
- contrastive embeddings (`z_rna`, `z_lig`)
- decoy separation logit / ranking score

Supervision:
- contrastive + ranking + decoy BCE/focal BCE

### C. What is exported for downstream tasks
After pretrain, keep these artifacts:
- encoder checkpoint (backbone weights)
- optional projection head checkpoint
- embedding cache (optional, for fast screening)

These are then reused by:
- affinity regression/ranking head
- virtual screening rank head
- binding-site per-residue head

## Generated manifests
- `gerna_affinity_manifest.csv`: GerNA-Bind supervised affinity rows
- `rnamigos_docking_manifest.csv`: RNAmigos2 affinity ranking rows
- `rnamigos_binary_manifest.csv`: RNAmigos2 affinity binary rows
- `rnamigos_pocket_nodes_manifest.csv`: RNAmigos2 pocket-node labels for binding-site pretraining
- `rnamigos_simulated_pretrain_manifest.csv`: simulated RNA 3D graphs for self-supervised pretraining
- `rnamigos_ligand_decoy_manifest.csv`: active/decoy simulated pairs for affinity pretraining
- `bridge_manifest.csv`: merged master manifest used by training scripts

## Recommended training flow (RNA-only)
1. Stage A: structure pretraining
- Use `rnamigos_simulated_pretrain_manifest.csv`
- Objective: RNA 3D tower pretraining + contrastive warmup

2. Stage B: affinity multi-task finetuning
- Mix `rnamigos_docking_manifest.csv` + `rnamigos_binary_manifest.csv`
- Objectives: contrastive + ranking + decoy separation (+ optional docking regression)

3. Stage C: binding-site finetuning
- Use `rnamigos_pocket_nodes_manifest.csv`
- Objective: per-node pocket probability + joint consistency with pair embedding space

## Downstream task recipes (explicit)

Use these task-specific configs for focused runs.

1. Virtual screening (ranking + decoy separation)
- Config: `configs/downstream_virtual_screening.yaml`
- Command:
```bash
python scripts/train_unified_multitask.py --config configs/downstream_virtual_screening.yaml
```
- Main objectives: contrastive + ranking + decoy BCE
- Suggested report metrics: EF1%, EF2%, NDCG, ROC-AUC/PR-AUC on active/decoy

2. Binding-site prediction (residue/node)
- Config: `configs/downstream_binding_site.yaml`
- Command:
```bash
python scripts/train_unified_multitask.py --config configs/downstream_binding_site.yaml
```
- Main objective: site BCE on RNA node logits
- Suggested report metrics: node AUROC, AUPRC, F1

3. Affinity prediction (regression + optional ranking)
- Config: `configs/downstream_affinity.yaml`
- Command:
```bash
python scripts/train_unified_multitask.py --config configs/downstream_affinity.yaml
```
- Main objectives: docking/affinity regression + optional ranking
- Suggested report metrics: RMSE, MAE, Pearson, Spearman

Internal Chinese task-level guidance:
- `docs_downstream_tasks_zh.md`

## What to improve next
1. Stronger split policy
- Use scaffold split for ligands and family split for RNA pockets to avoid leakage.

2. Label calibration
- Keep docking score normalization per pocket.
- Add temperature scaling for binary heads.

3. Better hard negatives
- Re-sample decoys by pocket geometry similarity, not only chemical property matching.

4. Multi-task balance
- Dynamic loss weighting (for ranking vs binary vs site losses) to prevent one task dominating training.

5. Transition readiness to protein-RNA
- Keep current schema fields stable (`target_id`, `target_structure_path`, `pocket_id`, `ligand_smiles`).
- Later add `partner_chain_type` and protein node features without breaking existing manifests.

## Decoy strategy (recommended)
If your goal is conference-grade evidence, decoy design quality is one of the highest-impact levers.

### 1) Increase decoy count in stages
- Start with 20 decoys per active for fast iteration.
- Move to 50 decoys per active for main experiments.
- Stress-test with 100 decoys per active for robustness.

Rationale:
- Too few decoys inflates ranking metrics.
- A staged schedule gives fast feedback first, then realistic difficulty.

### 2) Build hardness tiers
- Easy decoys: random chemistry-matched negatives.
- Medium decoys: similar physicochemical properties (MW, logP, HBA/HBD, TPSA).
- Hard decoys: high 2D similarity (ECFP/Tanimoto) but known non-binders.

Rationale:
- Hard negatives force the model to learn true interaction patterns instead of trivial chemistry shortcuts.

### 3) Use pocket-aware negative mixing
- In-batch negatives from other ligands in the same pocket family.
- Cross-pocket negatives from similar RNA geometry pockets.
- Keep a fixed fraction for each source (for example 50/30/20).

Rationale:
- Mixing local and global negatives improves both discrimination and OOD behavior.

### 4) Avoid leakage and optimistic bias
- Do not reuse near-duplicate ligands across train/test scaffolds.
- Keep RNA family split strict when reporting final numbers.
- Report results under all decoy regimes (20/50/100) rather than only the easiest one.

### 5) Minimal decoy ablation set
- Baseline: current decoy setup.
- +More decoys: 20 -> 50.
- +Hard decoys: property + similarity matched.
- +Pocket-aware negatives.
- Full decoy curriculum: all above combined.

## Can this be submitted to a conference?
Yes, this direction is publishable if you show clear gains under strict evaluation.

### Suggested target venues
- ISMB/ECCB (bioinformatics + ML).
- RECOMB (computational biology methods).
- NeurIPS/ICLR workshop tracks on ML for biology and drug discovery.

### Go/no-go criteria before submission
- Full model beats strong 3D supervised baseline on at least 2 independent split protocols.
- Gains hold when decoy count increases (no collapse at 50 or 100 decoys).
- OOD split (new scaffold + new RNA family) still shows measurable benefit.
- At least 3 seeds with stable variance and no single-seed effect.
- One clear mechanistic analysis (for example residue-level attribution aligning with known contacts).

### What reviewers will question first
- Is improvement from better decoys only, or from representation learning?
- Is there leakage across RNA families or ligand scaffolds?
- Do gains remain under hard decoys and larger screening libraries?

If you preempt these three points with ablations and strict splits, acceptance probability is much higher.

## Publication-grade experiment checklist
Use this checklist to make the story convincing and reviewer-resistant.

### A. Must-run baselines
Run all baselines with the same data protocol and report mean +/- std across seeds.

1. Sequence/2D only baseline
- No 3D inputs.

2. 3D supervised only baseline
- Use the same model family but no pretraining.

3. 3D + simulated pretraining (no decoy pretraining)
- Isolate the effect of graph pretraining.

4. 3D + ligand decoy pretraining (no graph pretraining)
- Isolate the effect of active/decoy pretraining.

5. Full model
- Graph pretraining + decoy pretraining + multitask finetuning.

### B. Split protocol (anti-leakage)
1. Ligand scaffold split
- Use Bemis-Murcko scaffold grouping.

2. RNA family/pocket split
- Separate homologous pockets across train/valid/test.

3. Combined hard split
- Hold out both unseen ligand scaffolds and unseen RNA families.

4. Report all three
- Random split can be included but never used as the only evidence.

### C. Metrics to report
1. Affinity classification
- ROC-AUC, PR-AUC, MCC, calibration error.

2. Affinity ranking
- EF1%, EF2%, NDCG, Spearman.

3. Binding site
- Residue/node AUROC, AUPRC, F1 at fixed threshold.

4. Reliability
- Mean +/- std over at least 3 random seeds.

### D. Core ablations
1. Remove simulated graph pretraining.
2. Remove decoy pretraining.
3. Freeze vs finetune pretrained encoder.
4. Single-task vs multitask training.
5. Pocket graph resolution sensitivity.
6. Predicted 3D vs experimental 3D comparison (when available).

### E. Suggested result table template
| Model | 3D | Sim-pretrain | Decoy-pretrain | Split | Affinity AUC | EF1% | Site AUPRC |
|---|---|---|---|---|---:|---:|---:|
| Seq/2D baseline | no | no | no | scaffold+family | - | - | - |
| 3D supervised | yes | no | no | scaffold+family | - | - | - |
| 3D + sim pretrain | yes | yes | no | scaffold+family | - | - | - |
| 3D + decoy pretrain | yes | no | yes | scaffold+family | - | - | - |
| Full model | yes | yes | yes | scaffold+family | - | - | - |

### F. Minimum acceptance criteria for internal go/no-go
1. Full model beats 3D supervised on at least 2 of 3 split settings.
2. Improvements are stable across seeds (no overlap or small overlap in std bands).
3. Binding-site performance does not drop while improving affinity.
4. At least one OOD setting shows clear gain.
