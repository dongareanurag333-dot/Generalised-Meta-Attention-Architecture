diff --git a/README.md b/README.md
index 5e8e264f778d120212eca4f5930fa10b65047a03..a6a558f3ebecf88044f3b8a30d89c5c6537941ed 100644
--- a/README.md
+++ b/README.md
@@ -84,50 +84,63 @@ Current Large Language Models:
 
 ### High-Level Pipeline
 
 ```
 
 Input Tokens
 ↓
 Base Model (Transformer / LLM)
 ↓
 MetaAttention Module
 ↓
 Meta-Representation (Global Reasoning State)
 ↓
 ├── Rule Induction Head
 ├── Epistemic Confidence Head (Know / Infer / Unsure)
 ├── Self-Critique Loop (Coherence Score)
 ↓
 Decision Layer
 ↓
 Answer / Revise / Abstain
 
 ```
 
 ---
 
+## ⚙️ Hardware-Inspired Reasoning View
+
+GMAA can also be understood through low-level computer architecture principles:
+
+- **Rule induction ↔ logic composition:** the rule head learns reusable decision patterns, analogous to composing simple boolean primitives (AND/OR/XOR) into richer circuits.
+- **Meta-attention ↔ control logic:** the global meta-query behaves like a control unit signal that coordinates which internal states are relevant for the next operation.
+- **Self-critique ↔ feedback paths:** coherence scoring introduces a feedback loop, similar to sequential logic where prior state influences the next state transition.
+- **Epistemic confidence ↔ status flags:** `Know / Infer / Unsure` resembles machine status bits that guide downstream branching (`accept / revise / abstain`).
+
+This analogy does **not** claim GMAA is digital logic in disguise; instead, it offers an engineering mental model for designing verifiable, stepwise reasoning behavior in neural systems.
+
+---
+
 ## 🔬 MetaAttention: Core Innovation
 
 ### Standard Self-Attention (Transformer)
 
 Each token attends to other tokens:
 
 \[
 \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
 \]
 
 This captures **syntactic and semantic relations**, but not reasoning quality.
 
 ---
 
 ### MetaAttention (GMAA)
 
 MetaAttention computes a **global query** from the model’s internal state:
 
 \[
 \bar{h} = \frac{1}{n}\sum_{i=1}^{n} h_i
 \]
 
 \[
 Q = W_q \bar{h}, \quad K = W_k H, \quad V = W_v H
 \]


---

```markdown
# Generalised MetaAttention Architecture (GMAA)

> **A Reasoning-First, Epistemically Aware Neural Architecture Beyond Transformers**

**Author & Owner:** Anurag Dongare  
**License:** Apache License 2.0  
**Project Type:** Research / Experimental  
**Domain:** NLP, Reasoning Systems, AI Safety, Post-Transformer Architectures  

---

## 📌 Executive Summary

The **Generalised MetaAttention Architecture (GMAA)** introduces a new paradigm for neural language models:  
**models that reason about their own reasoning**.

Traditional Transformer and LLM architectures excel at learning statistical relationships between tokens, but they lack:

- Explicit uncertainty awareness
- Self-critique mechanisms
- Rule induction capabilities
- The ability to abstain when unsure

GMAA addresses these limitations by introducing **MetaAttention**, an attention mechanism that operates over *reasoning paths instead of tokens*, coupled with epistemic and self-evaluative modules.

This repository provides:
- A **from-scratch implementation**
- **Streaming training on Common Crawl**
- **Hybrid integration with Large Language Models**
- **Evaluation and benchmarking tools**

---

## 🎯 Core Idea

> **Attention should not only relate tokens — it should evaluate beliefs.**

MetaAttention enables a model to:
- Form a **global reasoning state**
- Attend over internal representations from that state
- Judge *how confident it should be* in its outputs

This makes GMAA suitable for **trust-sensitive, safety-critical, and reasoning-heavy applications**.

---

## 🚀 Key Contributions

- 🔹 **MetaAttention Mechanism** (reasoning-path attention)
- 🔹 **Epistemic Confidence Modeling** (`Know / Infer / Unsure`)
- 🔹 **Self-Critique Loop** for incoherence detection
- 🔹 **Rule Induction Head** (neural-to-symbolic bridge)
- 🔹 **Hybrid LLM + MetaReasoner System**
- 🔹 **Streaming Foundation Training on Common Crawl (C4)**

This is **not a larger Transformer** — it is a **method-aware architecture**.

---

## 🧠 Motivation

Current Large Language Models:

| Limitation | Impact |
|----------|--------|
| No uncertainty modeling | Overconfident hallucinations |
| No self-critique | Unsafe outputs |
| Token-level reasoning only | Weak global reasoning |
| No abstention | Cannot say “I don’t know” |

**GMAA embeds epistemic reasoning directly into the architecture**, rather than patching it post hoc.

> *Transformers learn correlations.  
> MetaAttention learns credibility.*

---

## 🏗️ Architecture Overview

### High-Level Pipeline

```

Input Tokens
↓
Base Model (Transformer / LLM)
↓
MetaAttention Module
↓
Meta-Representation (Global Reasoning State)
↓
├── Rule Induction Head
├── Epistemic Confidence Head (Know / Infer / Unsure)
├── Self-Critique Loop (Coherence Score)
↓
Decision Layer
↓
Answer / Revise / Abstain

```

---

## 🔬 MetaAttention: Core Innovation

### Standard Self-Attention (Transformer)

Each token attends to other tokens:

\[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
\]

This captures **syntactic and semantic relations**, but not reasoning quality.

---

### MetaAttention (GMAA)

MetaAttention computes a **global query** from the model’s internal state:

\[
\bar{h} = \frac{1}{n}\sum_{i=1}^{n} h_i
\]

\[
Q = W_q \bar{h}, \quad K = W_k H, \quad V = W_v H
\]

\[
\text{MetaAttention}(H) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
\]

📌 **Key Difference:**  
The model attends *from understanding*, not from individual tokens.

---

## 🧩 Epistemic Modules

### 1. Epistemic Confidence Head
Classifies outputs into:
- **Know** – grounded factual knowledge
- **Infer** – probabilistic reasoning
- **Unsure** – insufficient evidence

### 2. Rule Induction Head
Projects neural representations into **rule-like vectors**, enabling:
- Pattern abstraction
- Consistency checks
- Symbolic reasoning interfaces

### 3. Self-Critique Loop
Outputs a coherence score indicating:
- Contradictions
- Hallucination likelihood
- Reasoning instability

---

## 🤝 Hybrid LLM + GMAA System

GMAA can operate as an **epistemic guardrail** over any LLM:

```

User Query
↓
LLM (Generator)
↓
Candidate Answer
↓
GMAA (Reasoning & Epistemic Evaluation)
↓
Accept / Revise / Abstain

````

### Benefits
- ↓ Hallucination rate
- ↑ Safety and trustworthiness
- Explicit uncertainty signaling
- Regulatory-friendly behavior

---

## 🔬 Training From Scratch

GMAA supports **foundation-style training** from random initialization.

### Dataset Support
- C4 (Common Crawl)
- OSCAR
- RedPajama (planned)

### Training Characteristics
- Streaming data (no full downloads)
- Language modeling objective
- Epistemic heads trained jointly
- Compatible with Google Colab

```bash
python train_common_crawl.py
````

---

## 📊 Representative Results (Observed)

| Capability             | Transformer | LLM    | Hybrid LLM + GMAA |
| ---------------------- | ----------- | ------ | ----------------- |
| Fluent Generation      | ✅           | ✅      | ✅                 |
| Hallucination Control  | ❌           | ❌      | ✅                 |
| Abstention Ability     | ❌           | ❌      | ✅                 |
| Epistemic Transparency | ❌           | ❌      | ✅                 |
| Safety-Critical QA     | Weak        | Medium | **Strong**        |

---

## 🧪 Use Cases

✔️ Scientific & medical QA
✔️ Legal reasoning
✔️ Safety-critical decision systems
✔️ Robotics & planning
✔️ Research assistants
✔️ Trustworthy chatbots

❌ Creative writing
❌ Entertainment-focused generation

---

## 📂 Repository Structure

```bash
.
├── marm.py                # Core MetaAttention Reasoning Model
├── meta_attention.py      # MetaAttention module
├── hybrid.py              # LLM + GMAA integration
├── train_common_crawl.py  # Streaming pretraining
├── evaluate.py            # Generation & evaluation
├── visualize/             # Architecture diagrams
├── experiments/           # Benchmarks
├── notebooks/             # Colab notebooks
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🛠️ Requirements

* Python ≥ 3.9
* PyTorch ≥ 2.0
* transformers
* datasets
* accelerate
* tqdm
* numpy

---

## 📜 License

Licensed under the **Apache License 2.0**.

You are free to:

* Use
* Modify
* Distribute
* Commercialize

**Attribution required.**

---

## 👤 Author

**Anurag Dongare**
Independent Researcher

Research Focus:

* Epistemic AI
* Reasoning Architectures
* AI Safety
* Post-Transformer Models

---

## 🧭 Roadmap

* [ ] Multi-layer MetaAttention
* [ ] Distributed training
* [ ] Formal epistemic guarantees
* [ ] Benchmark suite release
* [ ] Academic paper submission (NeurIPS / ICLR)

---

## 📖 Citation (Draft)

```bibtex
@misc{dongare2025gmaa,
  title={Generalised MetaAttention Architecture},
  author={Dongare, Anurag},
  year={2025},
  note={Research Preprint}
}
```

---

## ⭐ Final Statement

> **This project does not aim to sound intelligent.
> It aims to know when it is not.**


