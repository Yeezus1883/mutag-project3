---
layout: default
title: Beyond Accuracy — MUTAG GNN Project
---

# Beyond Accuracy: Building an Interpretable and Robust GNN for Molecular Mutagenicity Prediction

## Introduction



This project started as a standard **Graph Neural Network (GNN)** implementation on the **MUTAG dataset**, a classic benchmark for molecular mutagenicity prediction. But instead of stopping at classification performance, I explored a broader question:

> **How do we trust a graph model once it starts making high-confidence predictions?**

That led this project into a much more interesting direction — not just **prediction**, but also:

- **interpretability**
- **confidence calibration**
- **adversarial robustness**
- **local feature importance**
- **relevance decomposition**

The result is an interactive **Streamlit-based graph analysis app** built on top of a **GIN (Graph Isomorphism Network)** classifier, extended with several explainability and robustness experiments.

This write-up documents the full engineering and experimentation journey:
- what I built
- what worked
- what failed
- what I learned
- and how the project can be improved further

---

# 1. Problem Statement

The MUTAG dataset contains **molecular graphs**, where each graph represents a molecule and the task is to classify whether the molecule is **mutagenic** or **non-mutagenic**.

This is a natural graph learning problem because molecules are inherently structured as:

- **Nodes → atoms**
- **Edges → chemical bonds**
- **Graph label → mutagenicity class**

Traditional tabular ML methods lose a lot of structural information here. A **Graph Neural Network** is a much more appropriate model because it can learn from both:

- atom-level features
- relational structure between atoms

So the core problem was:

> **Can we train a GNN to classify mutagenic molecules accurately, and then meaningfully analyze why it makes those predictions?**

---

# 2. Why Graph Neural Networks for Molecules?

Molecules are not sequences and they are not images — they are **graphs**.

For example, a benzene ring or nitro group cannot be fully understood just from isolated atom features. The **connectivity pattern** matters just as much as the atoms themselves.

That makes GNNs especially suitable because they perform **message passing**, allowing each node to aggregate information from its neighbors.

For this project, I used a **Graph Isomorphism Network (GIN)** because it is one of the strongest graph classification architectures and is well-suited for capturing graph structure in relatively small datasets like MUTAG.

---

# 3. Project Architecture

The project has two major layers:

## A. Core ML Pipeline
This includes:
- loading and preprocessing MUTAG
- training a GIN model
- cross-validation
- evaluation
- saving the best model

## B. Analysis & Visualization Layer
This includes:
- molecule visualization
- node importance maps
- minimal subgraph extraction
- SHAP-style attribution
- saliency maps
- LRP-like relevance
- calibration analysis
- adversarial robustness testing

An interactive **Streamlit app** was built to expose these analyses visually.

---

# 4. Model Training

The model was trained using **10-fold cross-validation** to get a more reliable estimate of performance instead of relying on a single train-test split.

### Training Setup
- **Model**: GIN (Graph Isomorphism Network)
- **Framework**: PyTorch Geometric
- **Device**: Apple Silicon MPS / CPU fallback
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam
- **Evaluation**: Accuracy + classification metrics

The training pipeline was designed to:
- train on each fold
- aggregate predictions
- compute confusion matrix and metrics
- save the best-performing model

This formed the backbone for all later interpretability and robustness experiments.

---

# 5. Interactive App Features

To make the project more usable and demonstrable, I built a **Streamlit dashboard** with the following components:

- **Dataset Explorer**
- **Molecule Graph Viewer**
- **Prediction + Confidence Display**
- **Node Importance Toggle**
- **Minimal Subgraph Toggle**
- **SHAP Explanation Toggle**
- **Saliency Map Toggle**
- **LRP-like Relevance Toggle**
- **Calibration Tab**
- **Adversarial Robustness Tab**

This transformed the project from “just a training notebook” into a much more complete and inspectable ML system.

---

# 6. Task 1 — SHAP-based Feature Attribution for Molecular Graphs

## Objective

The goal of this task was to identify **which atoms contribute most strongly** to a mutagenicity prediction using **SHAP-style feature attribution**.

In theory, SHAP is attractive because it gives a principled explanation of feature contribution. However, applying SHAP to **graph-structured neural networks** is significantly harder than applying it to regular tabular models.

## What I Implemented

I attempted to build a **wrapper around the graph model** so that SHAP could operate on the node feature matrix and return atom-level relevance scores.

The intended pipeline was:

1. Flatten graph/node features into SHAP-compatible format
2. Pass them through a wrapper function
3. Use SHAP to estimate feature contributions
4. Aggregate SHAP scores into per-node importance
5. Visualize atom-level importance on the molecular graph

## What Worked

- SHAP could be integrated at a basic level
- The app successfully exposed a **SHAP explanation toggle**
- Some molecule-level attributions could be computed without crashing
- The explanation pipeline helped reveal the practical limitations of SHAP on graph models

## What Failed

This task exposed several implementation and conceptual issues:

### 1. Shape mismatch problems
Because SHAP expects relatively stable tensor shapes, applying it to graph data caused multiple issues such as:
- reshape errors
- mismatch between flattened and structured graph inputs
- incompatible expectations between SHAP and PyTorch Geometric

### 2. Graph structure is not naturally SHAP-friendly
SHAP works more naturally on:
- tabular features
- fixed-size inputs
- standard differentiable pipelines

But molecular graphs are:
- variable-sized
- structured
- relational
- dependent on `edge_index`, batching, and graph topology

That makes direct SHAP integration fragile.

### 3. Inconsistent visualization outputs
In several cases, SHAP explanations produced:
- all-white nodes
- near-zero scores
- unstable or non-intuitive attributions

This suggested that even when the code ran, the output was not always meaningful enough to trust directly.

## Key Lesson

> **SHAP is not impossible for graph models — but it is not plug-and-play.**

This experiment was valuable because it showed that **generic explainability tools often break down on graph neural networks** unless adapted carefully.

## How This Can Be Improved

Better future directions include:
- using **GNNExplainer**
- using **PGExplainer**
- using **Integrated Gradients**
- using graph-native attribution libraries instead of forcing SHAP onto graph inputs

## Verdict

### Status:
**Partially successful / exploratory**

### Value:
High learning value, moderate practical reliability.

---

# 7. Task 2 — Adversarial Robustness via Edge Perturbation

## Objective

The goal here was to test how robust the GIN model is when the **graph structure is corrupted**.

In molecular graphs, even small changes to connectivity can alter meaning significantly. So I wanted to answer:

> **How sensitive is the model to structural perturbations?**

## What I Implemented

I created a robustness testing pipeline where edges in test graphs were randomly perturbed at different magnitudes.

The perturbation setup included:
- **0% perturbation** (baseline)
- **5% perturbation**
- **10% perturbation**

For each perturbation level, the model was re-evaluated and its accuracy recorded.

## Results

The model showed a clear degradation trend:

- **0% perturbation → ~0.915 accuracy**
- **5% perturbation → ~0.840 accuracy**
- **10% perturbation → ~0.505 accuracy**

These results were plotted as a **robustness curve** inside the app.

## What Worked

- The perturbation pipeline worked cleanly
- Accuracy degradation was clearly measurable
- The robustness curve provided an intuitive visual summary
- This became one of the strongest “system-level” evaluations in the project

## What Failed / Limitations

### 1. Perturbations were random, not optimized
This was a **random attack**, not a true worst-case adversarial attack.

That means:
- it is useful for robustness testing
- but it does **not** represent the strongest possible attack

### 2. Chemical validity was not enforced
Some perturbed graphs may not correspond to chemically valid molecules.

So the experiment is best interpreted as:

> **graph-structural robustness**, not chemically realistic adversarial chemistry.

## Key Lesson

> **The model is highly dependent on graph connectivity and becomes brittle as structural noise increases.**

That is exactly what we would expect from a graph classifier — and it confirms that graph structure is doing real work in the decision-making process.

## How This Can Be Improved

Future improvements:
- PGD-style graph attacks
- chemically valid perturbation rules
- bond-type-aware attacks
- adversarial training

## Verdict

### Status:
**Successfully implemented**

### Value:
High practical value and strong demonstration of model stress testing.

---

# 8. Task 3 — Layer-wise Relevance Propagation (LRP) Decomposition

## Objective

The original task asked for **Layer-wise Relevance Propagation (LRP)** on the graph model.

Formal LRP for GNNs is non-trivial and would require:
- custom propagation rules
- deeper layer-wise decomposition logic
- graph-specific adaptation of standard LRP techniques

Instead of forcing a brittle or half-correct formal implementation, I implemented a more stable and practical alternative:

- **Saliency Maps**
- **Gradient × Input (LRP-like relevance)**

This gave a strong approximation of relevance decomposition without overcomplicating the model internals.

## What I Implemented

Two gradient-based explanation methods were added:

### A. Saliency Maps
This computes the gradient of the prediction with respect to the input node features.

Interpretation:
> “Which atoms does the model become most sensitive to if their features change?”

### B. Gradient × Input
This multiplies:
- input features
- by their gradients

Interpretation:
> “Which atoms actively contributed most strongly to the current prediction?”

This behaves similarly to a lightweight **LRP-style relevance approximation**.

## App Integration

The app was extended with:
- `Show Saliency Map`
- `Show LRP-like Relevance (Grad × Input)`

I also added a side-by-side comparison mode:
- **Saliency vs LRP-like relevance**

This was especially useful for comparing explanation behavior visually.

## What Worked

This ended up being one of the most useful explanation tasks in the project.

### Strong points:
- clean integration with the existing model
- stable execution
- meaningful per-node scores
- direct visual interpretability
- better practical usability than SHAP

In many cases, the highlighted atoms aligned better with chemically meaningful local structures than the earlier SHAP attempt.

## What Failed / Limitations

### 1. This is not “full formal LRP”
This is important to state honestly.

What was implemented is:
- **LRP-inspired**
- not a complete layer-by-layer formal LRP rule system

### 2. Gradient methods can be noisy
Like most gradient-based explanations, results can vary depending on:
- confidence saturation
- local sensitivity
- graph topology

So these should be interpreted as:
> useful local explanations, not absolute truth.

## Key Lesson

> **Gradient-based graph explanations are much more practical than trying to brute-force SHAP into a graph pipeline.**

This was one of the clearest “what works in practice vs what sounds good on paper” lessons in the project.

## How This Can Be Improved

Future upgrades:
- Integrated Gradients
- Captum-style graph attribution
- formal GNN-LRP implementation
- signed positive vs negative relevance separation

## Verdict

### Status:
**Successfully implemented (practical approximation)**

### Value:
Very high — strong interpretability gain with good engineering feasibility.

---

# 9. Task 4 — Cross-Fold Prediction Confidence Calibration

## Objective

Accuracy alone does not tell us whether a model’s **confidence scores are trustworthy**.

A model can be correct often and still be **badly calibrated** — for example:
- predicting 99% confidence too often
- being confidently wrong
- overestimating certainty on ambiguous molecules

So the goal here was:

> **Does the model’s confidence reflect its true reliability?**

## What I Implemented

I extended the evaluation pipeline to collect:
- predicted probabilities
- predicted labels
- true labels

From these, I computed:
- **ECE (Expected Calibration Error)**
- calibration curve data

The app also includes a dedicated **Calibration tab**.

## Results

A calibration curve was generated and the observed **ECE was approximately 0.0498**, which is relatively low.

This suggests that:

> **the model’s confidence scores are reasonably well aligned with actual correctness.**

## What Worked

- Calibration metrics were computed successfully
- ECE was integrated into the app
- The calibration curve was plotted correctly
- This added an important “trustworthiness” layer to the project

## What Failed / Limitations

### 1. Temperature scaling was not fully completed
The original advanced task suggested trying:
- temperature scaling
- calibration transfer across folds

These were explored conceptually, but the full deployment of a clean reusable scaling pipeline was not finalized.

### 2. Small dataset effects
MUTAG is a relatively small dataset, so calibration estimates can be somewhat noisy depending on fold composition.

## Key Lesson

> **A high-confidence graph model should not automatically be trusted — calibration gives that confidence meaning.**

This task added a very important practical ML dimension to the project.

## How This Can Be Improved

Future upgrades:
- temperature scaling
- isotonic regression
- fold-wise calibration transfer experiments
- reliability diagrams with confidence intervals

## Verdict

### Status:
**Successfully implemented**

### Value:
High — one of the strongest “trustworthiness” additions to the project.

---

# 10. Task 5 — Subgraph Masking & Local Feature Importance

## Objective

The goal here was to identify:

> **Which local parts of a molecule are most essential for the model’s prediction?**

This is one of the most intuitive explainability tasks for graph data because instead of just assigning scores, it asks:

> “What happens if I remove or suppress certain parts of the graph?”

## What I Implemented

I implemented a **node masking / local importance pipeline** that estimates how much the prediction changes when specific nodes or local regions are suppressed.

This was used to generate:
- **node importance scores**
- **minimal subgraph views**
- local explanation overlays inside the app

## What Worked

This ended up being one of the most visually interpretable parts of the project.

### Strong points:
- highlighted the most important atoms clearly
- supported minimal subgraph extraction
- worked well with graph visualization
- gave a more structural explanation than pure gradient methods

The app could show:
- important nodes in red
- less relevant nodes faded out
- the minimal predictive region of the graph

## Bug Encountered

A particularly interesting issue occurred here:

When the **minimal subgraph toggle** was enabled, some atoms (especially oxygen atoms in NO₂ groups) appeared to change into **carbon atoms**.

### Root Cause
This happened because:
- the graph structure was being visually filtered
- but atom labels were still being inferred from mismatched node indexing in the masked / displayed graph

### Fix
The issue was solved by:
- always reading labels from the original graph node features
- only fading / highlighting nodes based on importance
- avoiding accidental remapping of atom identity

This was a good example of how **interpretability tooling can fail visually even when the underlying model is fine**.

## Key Lesson

> **Subgraph-based explanation is often more intuitive for graph data than raw feature attribution.**

This task gave some of the most convincing “human-readable” explanations in the entire project.

## How This Can Be Improved

Future upgrades:
- edge masking
- bond importance
- chemically valid motif extraction
- automatic functional group discovery

## Verdict

### Status:
**Successfully implemented**

### Value:
Very high — arguably one of the strongest explanation components in the project.

---

# 11. What Worked Best Across the Whole Project

If I had to summarize the strongest components of the project, they would be:

## Most Successful
- **GIN training pipeline**
- **Node importance visualization**
- **Minimal subgraph extraction**
- **Calibration analysis**
- **Adversarial robustness testing**
- **Saliency / Grad × Input relevance**

## Most Educational but Least Stable
- **SHAP for graph models**

That split is actually useful.

Because one of the biggest lessons from this project was:

> **Some explainability methods sound better in theory than they behave in practice.**

---

# 12. What Failed, and Why That Was Still Useful

Not every failure here was a bad thing.

In fact, some of the most valuable learning came from things that **didn’t cleanly work**.

## Major Friction Points

### 1. SHAP + Graphs is messy
The biggest challenge in the project was adapting a generic explanation tool to graph-structured neural networks.

### 2. Formal LRP is non-trivial for GNNs
It is possible, but not lightweight.

### 3. Graph visualization can lie if indexing is mishandled
This showed how careful you need to be when interpreting explanations.

### 4. Small benchmark datasets limit certainty
MUTAG is great for experimentation, but it is still a small dataset.

These failures were useful because they pushed the project beyond:
> “just making the code run”

and into:
> “understanding where the method actually breaks.”

That is much more valuable.

---

# 13. Final Technical Takeaways

This project reinforced a few major ideas:

## 1. Accuracy is not enough
A model can be accurate but still:
- fragile
- overconfident
- uninterpretable

## 2. Explainability for graphs is harder than for tabular data
Graph structure introduces a lot of complexity in attribution.

## 3. Practical methods often beat theoretically elegant ones
Gradient-based relevance turned out to be more useful than SHAP in this context.

## 4. Robustness and calibration should be part of applied ML by default
Not just “nice to have” extras.

---

# 14. Future Improvements

There are several strong next steps for this project:

## Explainability
- GNNExplainer
- PGExplainer
- Integrated Gradients
- bond-level attribution

## Robustness
- PGD-style graph attacks
- chemically valid perturbation rules
- adversarial training

## Calibration
- temperature scaling
- isotonic regression
- fold-specific transfer analysis

## Deployment
- prediction from custom SMILES input
- API wrapper
- production-ready inference endpoint

---

# 15. Conclusion

This project started as a straightforward graph classification problem.

But the most interesting part was not simply making the model predict mutagenicity.

It was asking:

- **Why did it predict that?**
- **Can I trust the confidence?**
- **Will it still work if the graph changes?**
- **Which part of the molecule actually matters?**

That shift — from **accuracy to understanding** — is what made this project meaningful.

And in many ways, that is the real value of machine learning engineering:

> not just building a model that works,  
> but building one you can actually inspect, question, and trust.

---

# Appendix — Project Highlights

## Implemented
- GIN model training on MUTAG
- 10-fold cross-validation
- Prediction interface
- Node importance scoring
- Minimal subgraph extraction
- SHAP-style graph attribution
- Saliency maps
- Gradient × Input relevance
- Calibration analysis
- Adversarial perturbation robustness

## Built With
- Python
- PyTorch
- PyTorch Geometric
- Streamlit
- NetworkX
- Matplotlib
- NumPy
- scikit-learn
- SHAP

---

## Author

Built by **Eshaan** as part of a deeper exploration into:

- Graph Neural Networks
- Explainable AI
- Robust ML Systems
- AI for Scientific Applications