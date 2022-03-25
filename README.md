# Grimm BERT

This framework provides experiments for my master thesis about automatic dictionary generation.
Run [grimm_bert.py](/grimm_bert.py) to start the framework. Its `-h` option explains all possible arguments and default
values.

## Setup

You can use [grimm_env.yml](/grimm_env.yml)
to [create a conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
with all necessary python packages.

[grimm_bert.py](/grimm_bert.py) caches models and tokenizers in [/model_cache](/model_cache) by default.

## Tests

Run `python -m unittest` in the main directory to execute all tests in [/test](/test).

## Pipeline

We use a pre-trained [CharacterBERT](https://github.com/helboukkouri/character-bert) model calculate contextualized
word-vectors from input sentences. After this step with `pytorch`, we use `sklearn`, `numpy` and `pandas` to apply
metrics and transform data.
