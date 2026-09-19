# Prompt Wording Sensitivity in Small Language Models

Reproducible experiment scaffolding for testing whether semantically equivalent prompt wording changes the CommonsenseQA performance of `Qwen/Qwen3-1.7B`.

## Research design

Research question: **How sensitive is a small language model to semantically equivalent changes in prompt wording on commonsense reasoning tasks?**

- Model: `Qwen/Qwen3-1.7B`
- Dataset/split: `tau/commonsense_qa`, `validation`
- Final sample: 300 questions, selected through one seeded shuffle (`seed=42`)
- Pilot: first 10 questions from that same 300-question ordering
- Variants: five fixed instructions (`P1`–`P5`)
- Decoding: greedy (`do_sample=False`), `max_new_tokens=16`
- Qwen thinking: explicitly disabled with `enable_thinking=False`
- Constant conditions: the model, sampled questions, choices/question formatting, decoding configuration, answer extraction, and evaluation code are shared by every variant.

Greedy decoding is intentional: it minimizes generation randomness as a confounding variable when measuring sensitivity to prompt wording. The runner creates 50 generations in pilot mode (10 questions × 5 variants). It does not run anything unless invoked.

## Setup

Use Python 3.10+ and install a PyTorch build suitable for your hardware, then install the project packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the model and dataset through Hugging Face. Hardware and package versions can affect runtime; preserve `run_metadata.json` with every result set.

## Run the pilot

From the repository root:

```bash
PYTHONPATH=. python -m src.runner --pilot
```

Results are written to `outputs/pilot/`:

- `generations.jsonl`: one record per generation
- `run_metadata.json`: configuration, mode, expected generation count, timestamp, available Hugging Face dataset provenance (fingerprint/version/revision fields), and sampled question IDs
- `sampled_question_ids.json`: the exact sampled IDs, separately preserved for audit and reconstruction

Each generation record contains `question_id`, `prompt_variant`, `question`, `choices`, `ground_truth`, `raw_response`, `predicted_answer`, `correct`, `generated_tokens`, and `inference_time` (seconds).

To run the planned 300-question experiment only when authorized:

```bash
PYTHONPATH=. python -m src.runner --full
```

## Google Colab (T4 GPU)

When local storage is insufficient for the model weights, run the same pilot in Google Colab with a T4 GPU. Open [`notebooks/colab_pilot.ipynb`](notebooks/colab_pilot.ipynb) in Colab, select **Runtime → Change runtime type → T4 GPU**, set `REPO_URL` to your GitHub repository URL, and run the cells in order.

The notebook clones the repository, installs `requirements.txt`, verifies CUDA, runs `pytest`, executes **only** `python -m src.runner --pilot`, runs the existing analysis scripts, and downloads a zip of `outputs/pilot/`. It invokes the same centralized configuration and modules as a local run: seed 42, Qwen3-1.7B, thinking disabled, greedy decoding, and 16 generated tokens. It never invokes `--full`.

## Analyze completed runs

```bash
PYTHONPATH=. python scripts/analyze.py outputs/pilot/generations.jsonl
PYTHONPATH=. python scripts/plot_results.py outputs/pilot/generations.jsonl
```

These write summary CSVs, per-question prediction-disagreement data, and an accuracy plot. They never create results on their own.

## Answer extraction

`src/evaluation.py` accepts an exact A–E response first, then a clearly labelled answer (for example, `Answer: C`). As a conservative fallback it accepts a single standalone A–E letter. Contradictory explicit answer cues, ambiguous answers, or absent answers become `null` and are scored incorrect; raw text remains available for audit.

## Reproducibility and change control

The experiment settings are centralized in `src/config.py`; instructions are centralized in `src/prompts.py`. Do not modify the model, data split, sampling, prompts, decoding, or extraction/evaluation procedure between variants. Any scientifically meaningful departure from this design should be documented here and in the associated run metadata before executing it.
