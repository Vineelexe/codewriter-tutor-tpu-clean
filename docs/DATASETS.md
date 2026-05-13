# Datasets

Dataset handling is staged so the laptop does not need to hold huge processed datasets.
Raw and structured sources can live remotely, while final training shards are published as
a private Kaggle Dataset or remote archive for TPU consumption.

## Source Classes

- Python code corpora such as Stack v2 Python and CodeSearchNet Python.
- Structured debugging, testing, refactor, explanation, and tutoring records.
- Synthetic records generated offline by parent LLMs and validated before prepacking.
- Small local fixtures for tests and smoke runs.

## Splits and Leakage

Prepacking creates separate train, validation, and optional test outputs. Split assignment
must be deterministic and candidate-based where synthetic examples share a source candidate.
Validation loss must never depend on live JSONL streams or mutable raw sources.

## Storage

- Structured synthetic shards: private Hugging Face Dataset repo preferred.
- Final prepacked shards: private Kaggle Dataset and/or remote archive.
- Local laptop: tiny fixtures, manifests, reports, and smoke shards only.

## Mixture Balance

Every final shard must be shuffled and mixture-balanced so a TPU session sees stable source
and task distribution regardless of where it resumes. Loss weighting is applied before final
shard writing through sampling or oversampling; final `.npy` shards contain token IDs only.
