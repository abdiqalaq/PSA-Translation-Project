# Machine Translation of Public Service Announcements (PSAs)

A proof-of-concept multilingual machine translation system for Kenyan Public Service Announcements, translating between English and Kiswahili using few-shot cross-lingual transfer learning on pre-trained multilingual models. Developed as the semester project for **DSA 4020A: Natural Language Processing**.

> PSAs are short, action-oriented public information messages (e.g. *"Ministry of Health advises parents to take children for vaccination."*) used to inform, warn, or guide the public on health, safety, education, security, and governance matters.

## Team

- Abdiqalaq Issack
- Chiadika Elue
- Amy Njenga


**Course:** DSA 4020A Natural Language Processing — School of Science & Technology
**Supervisor:** Dr. Edward Ombui

## Project Status

| Week | Focus | Status |
|---|---|---|
| 1 | Data collection & curation | ✅ Complete |
| 2 | Data processing & EDA | ✅ Complete |
| 3 | Modeling with transfer learning | ✅ Complete |
| 4 | Evaluation & deployment | 🔄 In progress |

## Dataset

A synthetic English–Kiswahili parallel PSA corpus was generated using a knowledge-based, template-driven approach grounded in authentic PSAs from Kenyan government and public sources, then translated to Kiswahili using the NLLB-200 model.

| Item | Value |
|---|---|
| Total parallel sentence pairs | 50,000 |
| Domains | 5 (Health, Education, Agriculture, Security, Governance) |
| Sentences per domain | 10,000 |
| Missing / duplicate records | 0 |
| Train / Validation / Test split | 40,000 / 5,000 / 5,000 |

Dataset columns: `master_id`, `psa_id`, `domain`, `subcategory`, `psa_text`, `kiswahili_text`.

## Models & Results

Two pre-trained sequence-to-sequence models were fine-tuned for English → Kiswahili translation:

| Model | Training data | Test Loss | BLEU |
|---|---:|---:|---:|
| mT5-small (zero-shot baseline) | 0 | 25.2147 | 0.0210 |
| mT5-small (fine-tuned) | 40,000 examples | 0.4642 | 58.7627 |
| **NLLB-200 Distilled (fine-tuned)** | 10,000 examples | **0.1273** | **80.6249** |

NLLB-200 Distilled (600M) achieved the best performance, outperforming fine-tuned mT5-small despite using a quarter of the training data — indicating its multilingual pretraining transfers more effectively to this low-resource language pair. Validation BLEU (81.42) and test BLEU (80.62) are close, indicating stable generalization with minimal overfitting.

**Example translation (NLLB-200, unseen input):**
> EN: *"Wash your hands regularly to prevent disease."*
> SW: *"Osha mikono yako kwa ukawaida ili kuzuia magonjwa."*

### Training configuration

| Parameter | mT5-small | NLLB-200 Distilled (600M) |
|---|---|---|
| Epochs | 3 | 1 |
| Learning rate | 5e-5 | 2e-5 |
| Effective batch size | 16 | 16 |
| Max sequence length | 128 | 128 |
| Hardware | Tesla T4 GPU | Tesla T4 GPU |
| Experiment tracking | Weights & Biases | Weights & Biases |



## Tools & Stack

Python · Hugging Face `transformers` / `datasets` / `evaluate` · SacreBLEU · PyTorch · Weights & Biases · pandas · Google Colab (Tesla T4 GPU)

## Setup & Usage

```bash
git clone <repo-url>
cd <repo-name>
pip install transformers datasets evaluate sacrebleu sentencepiece accelerate wandb pandas
```

Open a notebook in `notebooks/` (Google Colab recommended for GPU access), mount the dataset from `data/`, and run cells sequentially. A Weights & Biases API key is required for experiment logging.

## Roadmap (Week 4)

- [ ] Additional automatic metrics (chrF, COMET)
- [ ] Human evaluation with native Kiswahili speakers (fluency, adequacy, cultural accuracy)
- [ ] Error analysis and limitations write-up
- [ ] Deployment as a Streamlit/Gradio web demo
- [ ] Extension to additional under-resourced languages (Ekegusii, Dholuo, Somali)

## License

This project is released under the [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
