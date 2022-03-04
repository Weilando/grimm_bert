# Grimm BERT

This framework provides experiments for my master thesis about automatic dictionary generation.

## Setup

You can use [grimm_env.yml](/grimm_env.yml) to [create a conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) with all necessary python packages.

Run `python download_bert.py` to cache a HuggingFace model in [/models](/models).
Its option `-h` explains additional arguments, e.g., for different models.

## Tests

Call `python -m unittest` in the main directory to execute all tests in [/test](/test).

## Pipeline

We use a pre-trained BERT model from `transformers` (Huggingface) and its according tokenizer to retrieve tokens and contextualized word-vectors from input sentences.
After this step with `pytorch`, we use `sklearn`, `numpy` and `pandas` to apply metrics and transform data.
