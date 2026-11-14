# BoxOfficeIQ — DistilBERT Sentiment Analysis Methodology

> **Version:** 1.0  
> **Model:** DistilBERT (`distilbert-base-uncased`) fine-tuned for Binary Sentiment Classification  
> **Target Task:** Movie Review Sentiment & Audience Reception Scoring

---

## Table of Contents

1. [Why DistilBERT?](#why-distilbert)
2. [Data & Fine-Tuning Process](#data--fine-tuning-process)
3. [Tokenization & Input Representation](#tokenization--input-representation)
4. [Label Mapping & Softmax Probabilities](#label-mapping--softmax-probabilities)
5. [Evaluation Metrics Explained](#evaluation-metrics-explained)
6. [How to Interpret Sentiment Scores](#how-to-interpret-sentiment-scores)
7. [System Integration with Forecasting](#system-integration-with-forecasting)
8. [Limitations & Practical Considerations](#limitations--practical-considerations)

---

## Why DistilBERT?

Modern NLP classification tasks benefit significantly from transformer architectures. However, full-scale models (such as BERT-base or RoBERTa) introduce significant computational overhead, memory footprints, and latency during inference.

BoxOfficeIQ adopts **DistilBERT** (`distilbert-base-uncased`):

1. **Lightweight & Fast:** DistilBERT retains **97% of BERT's language understanding** capacity while being **40% smaller** (66M parameters vs. 110M) and **60% faster** during CPU/GPU inference.
2. **Knowledge Distillation:** Trained using student-teacher distillation (Hinton et al.), where the smaller student network mimics both the hard labels and the soft probability distributions learned by full BERT.
3. **Pre-trained on English Corpus:** Extensive pre-training on Toronto BookCorpus and English Wikipedia allows it to handle nuanced film reviews, sarcastic tones, idioms, and sentiment vocabulary out of the box.
4. **Low Inference Latency:** Crucial for near real-time review classification in FastAPI backend services without requiring dedicated high-end GPU clusters.

---

## Data & Fine-Tuning Process

### Dataset & Preprocessing
The sentiment model is trained and fine-tuned on movie reviews (`ml/data/raw/sample_reviews.csv` or benchmark IMDb datasets):
- **Sample size:** 500 stratified film reviews (or full 25k/50k benchmark sets).
- **Split:** 80% Training, 20% Validation/Test (stratified by class label).
- **Text cleanup:** Standard whitespace normalization, removal of HTML tags (e.g. `<br />`), with casing preserved for uncased tokenization.

### Hyperparameters & Optimization

| Parameter | Configuration | Justification |
|---|---|---|
| **Base Model** | `distilbert-base-uncased` | Standard English transformer backbone |
| **Epochs** | 3 | Prevents catastrophic forgetting & overfitting |
| **Optimizer** | `AdamW` (Weight Decay Fix) | Weight decay prevents large weights on rare words |
| **Learning Rate** | `2e-5` (with linear warmup) | Standard fine-tuning learning rate for transformers |
| **Batch Size** | 16 (or 32 depending on VRAM) | Balanced gradient stability |
| **Weight Decay** | 0.01 | L2 regularization |
| **Loss Function** | Cross-Entropy Loss | Standard multi-class / binary classification loss |

```python
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
```

---

## Tokenization & Input Representation

- **Tokenizer:** `DistilBertTokenizerFast` (WordPiece algorithm).
- **Max Sequence Length:** 256 tokens.
  - *Rationale:* Over 90% of film audience reviews fall well below 256 sub-word tokens. Truncating at 256 eliminates unnecessary padding computations while preserving the salient opinionated content (typically found in the opening and closing sentences).
- **Special Tokens:**
  - `[CLS]` (Classification Token): Preceded at sequence start; its pooled hidden state serves as aggregate representation.
  - `[SEP]` (Separator Token): Marks sentence boundary.
  - `[PAD]` (Padding Token): Padded dynamically to the longest sequence in the batch.

```python
inputs = tokenizer(
    text,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)
```

---

## Label Mapping & Softmax Probabilities

The classification head outputs 2 logits corresponding to class indices:

| Logit Index | Label String | Interpretation |
|---|---|---|
| `0` | `negative` | Critical, dissatisfied, low audience reception |
| `1` | `positive` | Favourable, enthusiastic, high recommendation |

### Probability Calculation
Logits `z = [z₀, z₁]` are passed through a Softmax function:

```
P(positive) = exp(z₁) / (exp(z₀) + exp(z₁))
P(negative) = exp(z₀) / (exp(z₀) + exp(z₁))
```

- **Predicted Label:** `argmax([P(negative), P(positive)])`
- **Confidence Score:** `max(P(negative), P(positive))`

---

## Evaluation Metrics Explained

Model performance is evaluated on the held-out test split using:

### 1. Confusion Matrix
```
                  Predicted Negative    Predicted Positive
Actual Negative          TN                    FP
Actual Positive          FN                    TP
```

### 2. Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Overall percentage of correct predictions. Useful when test classes are balanced.

### 3. Precision (Positive Class)
```
Precision = TP / (TP + FP)
```
Measures accuracy among reviews predicted as positive. High precision ensures that movies flagged with good word-of-mouth genuinely have favourable audience sentiment.

### 4. Recall (Positive Class)
```
Recall = TP / (TP + FN)
```
Measures the fraction of actually positive reviews correctly caught by the model.

### 5. F1-Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
Harmonic mean of precision and recall. Provides a unified metric balancing false alarms and missed positive reactions.

---

## How to Interpret Sentiment Scores

In the BoxOfficeIQ interface and APIs, sentiment is exposed at two levels:

### 1. Individual Review Level
- `predicted_label`: `"positive"` or `"negative"`
- `confidence`: Range `[0.50, 1.00]`.
  - `> 0.90`: Strong, unambiguous sentiment ("Absolute masterpiece!").
  - `0.60 – 0.80`: Moderate sentiment with minor mixed commentary.
  - `0.50 – 0.60`: Neutral or ambiguous review borderline decision.

### 2. Movie-Level Aggregation (`positive_ratio`)
```
Positive Ratio = Positive Reviews Count / Total Analyzed Reviews
```
- **> 75% Positive:** Excellent word-of-mouth. Anticipate stronger-than-baseline second-weekend holdovers.
- **50% – 75% Positive:** Moderate reception. Standard drop-off curve expected.
- **< 50% Positive:** Poor word-of-mouth. Anticipate steep attendance decay (typically > 55% week-over-week drop).

---

## System Integration with Forecasting

While the primary attendance forecasting engine uses gradient boosting (XGBoost/LightGBM) on temporal and lag indicators, the sentiment engine provides **early warning signals**:
1. **Holding Multiplier:** If `positive_ratio > 0.80`, cinema managers can adjust the simulator's scenario modifier or extend screening slots.
2. **Cancellation Risk:** A sudden influx of negative reviews during opening weekend signals immediate demand softening before historical attendance lags catch up.

---

## Limitations & Practical Considerations

1. **Sentiment ≠ Immediate Ticket Demand:**
   - A cult arthouse film might have 95% positive reviews but low commercial attendance due to niche appeal.
   - A star-studded commercial tentpole may receive 45% negative critic reviews yet break attendance records on opening weekend.
2. **Review Volume Bias:**
   - Popular blockbusters receive hundreds of reviews within 48 hours; regional or smaller films may only have 5–10 reviews, making the aggregate ratio noisy.
3. **Sarcasm and Colloquial Language:**
   - Film reviews frequently employ sarcasm ("The best part was when the credits rolled"), which can occasionally mislead sub-word transformer classifiers.
4. **Review Authenticity:**
   - Synthetic or scraped sample reviews may not reflect post-release audience verification filters present on official platforms.
