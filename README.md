# Grimm BERT

[grimm_bert.py](/grimm_bert.py) provides pipelines for my master thesis about Automatic Dictionary Generation. Its `-h`
option explains all possible arguments and default values.

## Setup

### Conda Environment

You can use [grimm_env.yml](/grimm_env.yml)
to [create a conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
with all required python packages.

### Corpora

Use the corresponding pre-processor to generate suitable input files for the pipeline.
The input files from WSDEval need to be in [data/wsdeval_corpora](/data/wsdeval_corpora) and raw text corpora
in [data/raw_text_corpora](/data/raw_text_corpora).

| Corpus      | Description                                                  | Pre-Processor                                              |
|-------------|--------------------------------------------------------------|------------------------------------------------------------|
| Toy         | Simple corpus for small tests.                               | [data.ToyPreprocessor](/data/toy_preprocessor.py)          |
| WSDEval     | Evaluation corpora from SemEval 2007/13/15 and Senseval 2/3. | [data.WsdevalPreprocessor](/data/wsdeval_preprocessor.py)  |
| SemCor      | Semantic concordance with more than 800k tokens.             | [data.WsdevalPreprocessor](/data/wsdeval_preprocessor.py)  |
| Shakespeare | Shakespeare's works in raw text.                             | [data.RawTextPreprocessor](/data/raw_text_preprocessor.py) |

Please use [data.WsdevalPreprocessor](/data/wsdeval_preprocessor.py) to add a corpus in WSDEval's XML-format
or [data.RawTextPreprocessor](/data/raw_text_preprocessor.py) for a raw text corpus.
If both do not apply, create a new subclass of [data.CorpusPreprocessor](/data/corpus_preprocessor.py).

### System Requirements

Our annotated Toy corpus with three sentences can be executed on a standard computer, e.g., with a dual-core CPU and 8GB
RAM. For the SemCor corpus, we recommend 16GB RAM and one CPU. Please note that you mainly benefit from a multicore CPU
or GPU while calculating word vectors with CharacterBERT. As the first pipeline run caches the word vectors for reuse,
further runs do not need multiple cores.

## Pipelines

We take a list of sentences as input, where each sentence is a list of tokens (of type `str`).

1. Lower case all sentences.
2. Wrap each sentence with special tokens `[CLS]` and `[SEP]`.
3. Calculate one contextualized word vector per token with a
   pre-trained [CharacterBERT](https://github.com/helboukkouri/character-bert) model.
4. Collect all corresponding word vectors and references per token.
5. Perform Word Sense Discrimination per token with hierarchical clustering.

Depending on the command line arguments, the clustering uses different criteria to cut the dendrogram.
If several arguments are given, the pipeline performs the highest criterion from the table.
We recommend to use the known senses from the ground truth, if given.
Otherwise, choosing a maximum distance is most promising.
Good starting ranges are 8-10 for Euclidean distances.

| Argument             | Description                                                            |
|----------------------|------------------------------------------------------------------------|
| `--known_senses`     | fit the number of senses from the ground truth                         |
| `--max_distance d`   | cuts each dendrogram at a given maximum linkage distance               |
| `--min_silhouette s` | predict the number of senses with the Silhouette Coefficient criterion |

## Evaluation

- Our [evaluation notebook](/evaluation.ipynb) offers many plots and statistics that deliver insights like sense counts
  and clustering metrics.
- On the one hand, you can browse the generated dictionary in the last section of this notebook as a **DataFrame**. On
  the other hand, you can use the notebook or the `--export_html` pipeline option to generate an **HTML page** with the
  dictionary and corresponding sentences from the training corpus.

## Caches

The software uses caches to enable executions in offline HPC environments and to speed up repeated calculations.

- Models and tokenizers: [model_cache](/model_cache)
- Corpora: [data/corpus_cache](/data/corpus_cache)
- Word vector matrix and raw `id_map` per corpus: user defined result location

## Tests

Run `python -m unittest` in the main directory to execute all tests in [test](/test).
