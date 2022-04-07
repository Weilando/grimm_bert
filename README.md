# Grimm BERT

[grimm_bert.py](/grimm_bert.py) provides a pipeline for my master thesis about automatic dictionary generation. Its `-h`
option explains all possible arguments and default values.

## Setup

You can use [grimm_env.yml](/grimm_env.yml)
to [create a conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
with all necessary python packages.

Use [data.ToyPreprocessor](/data/toy_preprocessor.py) or [data.SemcorPreprocessor](/data/semcor_preprocessor.py) to
generate suitable input files for the pipeline. To add a new corpus, create a subclass
of [/data/CorpusPreprocessor](/data/corpus_preprocessor.py). You might want to tokenize sentences with an uncased
tokenizer, e.g., `BertTokenizer.basic_tokenizer`.

## Pipeline

We take a list of sentences as input, where each sentence is list of str tokens.

1. Lower case all sentences.
2. Wrap each sentence with special tokens `[CLS]` and `[SEP]`.
3. Calculate one contextualized word vector per token with a
   pre-trained [CharacterBERT](https://github.com/helboukkouri/character-bert) model.
4. Group all corresponding word vectors and references per token.
5. Perform word sense disambiguation per token with hierarchical clustering based on cosine similarities.
6. Evaluate the dictionary with the adjusted rand index (ARI).

If no maximum distance is given, the pipeline uses the number of senses from the ground truth to cluster word vectors.
Otherwise, it uses the maximum distance to distinguish different senses per token.

## Caches

The software uses caches to enable executions in offline HPC environments and to speed up repeated calculations.

- Models and tokenizers: [/model_cache](/model_cache)
- Corpora: [/data/corpus_cache](/data/corpus_cache)
- Word vector matrix and raw `id_map` per corpus: user defined result location

## System Requirements

Our annotated Toy corpus with three sentences can be executed on a standard computer, e.g., with a dual-core CPU and 8GB
RAM. For the SemCor corpus, we recommend 16GB RAM and one CPU. Please note that you mainly benefit from a multicore CPU
or GPU while calculating word vectors with CharacterBERT. As the first pipeline run caches the word vectors for reuse,
further runs do not need multiple cores.

## Tests

Run `python -m unittest` in the main directory to execute all tests in [/test](/test).
