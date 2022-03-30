# Grimm BERT

[grimm_bert.py](/grimm_bert.py) provides a pipeline for my master thesis about automatic dictionary generation. Its `-h`
option explains all possible arguments and default values.

## Setup

You can use [grimm_env.yml](/grimm_env.yml)
to [create a conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
with all necessary python packages.

## Pipeline

We take a list of sentences as input, where a sentence can either be a string or a list of tokens.

1. Tokenize sentences, if necessary.
2. Lower case all sentences.
3. Wrap each sentence with special tokens `[CLS]` and `[SEP]`.
4. Calculate one contextualized word vector per token with a
   pre-trained [CharacterBERT](https://github.com/helboukkouri/character-bert) model.
5. Group all corresponding word vectors and references per token.
6. Perform word sense disambiguation per token with hierarchical clustering based on cosine similarities.
7. Evaluate the dictionary with the adjusted rand index (ARI).

## Caches

The software uses caches to enable executions in offline HPC environments and to speed up repeated calculations.

- Models and tokenizers: [/model_cache](/model_cache)
- Corpora: [/data/corpora](/data/corpora)
- Word vector matrix and raw `id_map` per corpus: user defined result location

## System Requirements

Our annotated Toy corpus with three sentences can be executed on a standard computer, e.g., with a dual-core CPU and 8GB
RAM. For the SemCor corpus, we recommend 16GB RAM and one CPU. Please note that you mainly benefit from a multicore CPU
or GPU while calculating word vectors with CharacterBERT. As the first pipeline run caches the word vectors for reuse,
further runs do not need multiple cores.

## Tests

Run `python -m unittest` in the main directory to execute all tests in [/test](/test).
