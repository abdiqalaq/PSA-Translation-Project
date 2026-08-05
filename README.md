# English–Ekegusii Neural Machine Translation for Public Service Announcements

A neural machine translation (NMT) system that translates Public Service Announcements (PSAs) from English into Ekegusii, a low-resource Bantu language spoken in Kenya, using multilingual transfer learning. Developed as the semester project for **DSA 4020A: Natural Language Processing**, University of Nairobi.

> PSAs are short, action-oriented public information messages (e.g. *"Residents urged to maintain clean environments."*) used to inform, warn, or guide the public on health, safety, education, security, and governance matters. Most PSAs in Kenya are published only in English and Kiswahili, leaving speakers of indigenous languages like Ekegusii without access.

## Team

- Abdiqalaq Issack
- Amy Njenga
- Chiadika Elue
- Joyann Maina

**Course:** DSA 4020A Natural Language Processing — School of Science & Technology
**Supervisor:** Dr. Edward Ombui

## Problem Statement

Ekegusii lacks sufficiently large parallel corpora to train high-quality neural translation systems directly, and existing multilingual translation models perform poorly on English–Ekegusii translation without adaptation. This project addresses that gap using transfer learning from a related, higher-resource Bantu language (Kiswahili) combined with a custom-adapted multilingual model.

## Objectives

1. Collect a large multilingual PSA corpus (English–Kiswahili).
2. Develop a high-quality, manually validated Gold English–Ekegusii dataset.
3. Investigate transfer learning from English–Kiswahili to English–Ekegusii.
4. Compare mT5 and NLLB-200 architectures for low-resource translation.
5. Evaluate translation quality using automatic and human evaluation.
6. Produce a deployable translation prototype.

## Approach

**Stage 1 — English–Kiswahili Corpus:** ~50,000 PSA sentence pairs collected from public sources (Ministry of Health, WHO, Kenya Red Cross, government websites, NGOs, and agriculture/education institutions) across five domains: health, education, agriculture, governance, and security.

**Stage 2 — Gold English–Ekegusii Dataset:** A manually curated and validated dataset of 3,875 English–Ekegusii sentence pairs, split into training (3,100), validation (387), and test (388) sets.

**Transfer Learning:** An mT5 model was first fine-tuned on the large English–Kiswahili corpus. Because Kiswahili and Ekegusii are both Bantu languages, this learned representation was transferred to initialize English–Ekegusii fine-tuning on the much smaller Gold dataset.

**NLLB-200 Adaptation:** NLLB-200 Distilled does not natively support Ekegusii. It was adapted by adding a custom language token (`guz_Latn`), extending the tokenizer vocabulary, resizing model embeddings, and fine-tuning on the Gold dataset.

## Results

Automatic evaluation on the held-out test set:

| Model | BLEU ↑ | chrF ↑ | COMET ↑ |
|---|---:|---:|---:|
| mT5 Baseline (English→Ekegusii, direct) | 0.14 | 2.46 | — |
| **NLLB-200 + `guz_Latn` (adapted)** | **4.12** | **33.16** | **0.601** |

The adapted NLLB-200 model outperformed the direct mT5 baseline across every metric. Absolute scores are modest, reflecting the difficulty of translation with under 4,000 training sentences, but the consistent improvement demonstrates that multilingual transfer learning meaningfully helps low-resource translation. Both BLEU/chrF (lexical overlap) and COMET (semantic quality) were used since correct translations can vary in wording, especially in low-resource settings.

Human evaluation was conducted by native Ekegusii speakers using a five-point Likert scale across three criteria: fluency, adequacy, and cultural accuracy.

**Example translation (NLLB-200):**
> EN: *"Residents urged to maintain clean environments."*
> Gold reference: *"Abanyaabamenerigwe bokobwata endagano y'okorabe oborogo."*
> NLLB prediction: *"Abanya bamenyerigwe bokobwata endagano y'okorabe oborogo."*



## Tools & Stack

Python · PyTorch · Hugging Face Transformers & Datasets · SentencePiece · Evaluate · COMET · Weights & Biases · NVIDIA A100-SXM4-80GB GPU (Kinesis GPU platform)

## Setup & Usage

```bash
git clone https://github.com/your-repo/english-ekegusii-nmt
cd english-ekegusii-nmt
pip install transformers datasets evaluate sacrebleu unbabel-comet sentencepiece accelerate wandb pandas
```

Run notebooks in `notebooks/` in order (a GPU is strongly recommended — an A100 or equivalent). A Weights & Biases API key is required for experiment logging.

## Limitations

- Automatic metrics are based on a small (388-sentence) test set typical of low-resource evaluation.
- The Gold dataset (3,875 pairs) is small relative to typical NMT training corpora; scores are expected to improve substantially with more data.
- Evaluation covers English→Ekegusii only; the reverse direction was not tested.
- Full-scale human evaluation (beyond the initial native-speaker review) has not yet been conducted.

## Future Work

- Expand the Gold English–Ekegusii corpus
- Incorporate carefully reviewed synthetic data
- Conduct larger-scale human evaluation
- Improve domain-specific terminology handling
- Deploy as a Streamlit/Gradio web application
- Extend the approach to other Kenyan indigenous languages



