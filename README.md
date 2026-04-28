# Neural Constituency Parsing for Yiddish
### via Vocabulary Adaptation and Parameter-Efficient Fine-Tuning

The first neural constituency parser for Yiddish, built using a 4-stage domain-adaptive pipeline over XLM-RoBERTa. Developed as part of the NLP course, Tel Aviv University, 2026.

---

## Background

Yiddish is a low-resource language with significant historical orthographic variation. While prior work established the first POS tagger for Yiddish (Kulick et al., 2022), no syntactic parser existed. Standard multilingual models like XLM-R suffer from severe **subword fragmentation** when processing Hebrew-script Yiddish — a problem we call the *"Tokenization Tax"*.

This project directly addresses that gap.

---

## Pipeline

```
JOCHRE corpus
     │
     ▼
 Top 2,000 Hebrew-script tokens extracted
     │
     ▼
 FOCUS vocabulary injection into XLM-R (Yiddish-pretrained)
     │
     ▼
 Domain-Adaptive Pre-training (MLM) on 5.5M YBC sentences
     │
     ▼
 CRF Constituency Parser trained on PPCHY (~200k words)
```

The entire XLM-R backbone (278M parameters) is **frozen**. Only the ~8M parameter parser head is trained, using **scalar mixing** across all 12 encoder layers to extract rich syntactic representations — keeping total trainable parameters under 100M.

---

## Results

| Model | Labeled F1 | Unlabeled F1 | Labeled Complete Match |
|---|---|---|---|
| Baseline (raw XLM-R + scalar mixing) | **83.81%** | **90.83%** | **50.93%** |
| Adapted (FOCUS + DAPT) | 83.02% | 90.16% | 49.42% |

The baseline correctly parsed **more than half of all test sentences** in their entirety (LCM = 50.93%), establishing a strong first neural baseline for Yiddish syntax.

---

## Key Finding: The "Zombie Token" Phenomenon

After FOCUS injection, all 1,875 new tokens existed in the embedding matrix but were **never retrieved** by the model — 0 emissions, 0% coverage. We term these *Zombie Tokens*.

Domain-Adaptive Pre-training (DAPT) via Masked Language Modeling progressively resolved this:

| Stage | Injected Token Emissions | Coverage |
|---|---|---|
| Post-injection (pre-DAPT) | 0 | 0.00% |
| Mid-training (~step 6,000) | 3,218 | 31.04% |
| Final checkpoint (step 12,000) | 4,713 | **43.73%** |

Despite successful token activation, downstream parsing performance remained statistically equivalent to the baseline. This suggests that constituency parsing is **resilient to orthographic noise** — the frozen XLM-R backbone already captures sufficient structural signal from fragmented native subwords.

> Mathematical initialization (FOCUS) is necessary but not sufficient for token activation. And for frozen-encoder parsing, solving the orthographic problem does not introduce novel syntactic signal.

---

## Data

| Corpus | Role |
|---|---|
| [JOCHRE](https://gitlab.com/jochre/corpora/jochre-yiddish-corpus) (Urieli, 2025) | Vocabulary source — top 2,000 clean Hebrew-script tokens |
| [Yiddish Book Center](https://www.yiddishbookcenter.org/) (YBC, 2023) | DAPT pre-training — 5.5M sentences (~100M tokens) |
| [PPCHY](https://github.com/skulick/ppchyprep) (Kulick et al., 2022) | Parsing — gold-standard constituency trees, 15th–20th century Yiddish |

---

## Future Work

- **Targeted unfreezing** of upper transformer layers to allow injected tokens to influence deeper contextual representations
- **Full fine-tuning** of the backbone without parameter constraints
- **Modern Yiddish coverage** — the PPCHY covers historical text; modern dialect parsing remains an open problem
- **Dependency parsing** as an alternative syntactic formalism

---

## References

- Dobler & de Melo (2023). [FOCUS: Effective embedding initialization for monolingual specialization of multilingual models.](https://aclanthology.org/2023.emnlp-main.829/) *EMNLP 2023.*
- Kulick, Ryant & Wallenberg (2022). A part-of-speech tagger for Yiddish. *LREC 2022.*
- Urieli (2025). Jochre Yiddish Corpus.
- Zhang et al. (2020). Fast and accurate neural CRF constituency parsing (SuPar). *ACL 2020.*

---

## Acknowledgments

This work was supported by the NLP course, Tel Aviv University, 2026.  
AI tools (Gemini) were used for assistance with preprocessing scripts and LaTeX formatting. All research design, methodology, and analysis were carried out by the authors.
