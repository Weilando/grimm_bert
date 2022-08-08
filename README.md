# Grimm BERT

[grimm_bert.py](/grimm_bert.py) provides pipelines for my master thesis about Automatic Dictionary Generation. Its `-h`
option explains all possible arguments and default values.

To get started, we recommend to run pipelines with the Euclidean linkage distance, the Average linkage criterion and a
pre-trained general [CharacterBERT](https://github.com/helboukkouri/character-bert) model. 8-10 is a good search space
for the distance threshold in the beginning. Let us give you an example call:

```
python grimm_bert.py first_experiments/Senseval2_d8 Senseval2 Euclidean Average -l INFO -d 8.0 -m './model_cache/general_character_bert'
```

## Setup

### Conda Environment

Use [grimm_env.yml](/grimm_env.yml) to create a conda environment with all required python packages.

### Corpora

Use the corresponding pre-processor to generate suitable input files for the pipeline. The input files from
[WSDEval](http://lcl.uniroma1.it/wsdeval/) need to be in [data/wsdeval_corpora](/data/wsdeval_corpora) and raw text
corpora in [data/raw_text_corpora](/data/raw_text_corpora).
[UFSAC](https://github.com/getalp/UFSAC) provides additional compatible corpora and extends WSDEval.

| Corpus      | Description                                     | Pre-Processor                                              |
|-------------|-------------------------------------------------|------------------------------------------------------------|
| Toy         | Simple corpus for small tests                   | [data.ToyPreprocessor](/data/toy_preprocessor.py)          |
| SemEval2007 | Evaluation corpus from SemEval 2007, Task 17    | [data.WSDEvalPreprocessor](/data/wsdeval_preprocessor.py)  |
| SemEval2013 | Evaluation corpus from SemEval 2013, Task 12    | [data.WSDEvalPreprocessor](/data/wsdeval_preprocessor.py)  |
| SemEval2015 | Evaluation corpus from SemEval 2015, Task 13    | [data.WSDEvalPreprocessor](/data/wsdeval_preprocessor.py)  |
| Senseval2   | All-Words task from Senseval 2                  | [data.WSDEvalPreprocessor](/data/wsdeval_preprocessor.py)  |
| Senseval3   | All-Words task from Senseval 3                  | [data.WSDEvalPreprocessor](/data/wsdeval_preprocessor.py)  |
| SemCor      | Semantic concordance (>800k tokens)             | [data.WSDEvalPreprocessor](/data/wsdeval_preprocessor.py)  |
| Shakespeare | Shakespeare's works in raw text (>1,15M tokens) | [data.RawTextPreprocessor](/data/raw_text_preprocessor.py) |

To add a new corpus, use [data.WSDEvalPreprocessor](/data/wsdeval_preprocessor.py) for a corpus in the WSDEval
XML-format and [data.RawTextPreprocessor](/data/raw_text_preprocessor.py) for a raw text corpus.
If both do not apply, create a new subclass of [data.CorpusPreprocessor](/data/corpus_preprocessor.py).

### Models

We support models with the [CharacterBERT](https://github.com/helboukkouri/character-bert) model architecture. The
command line argument `--model_cache` specifies the model weights.

To add new model architectures, you need to add its name to [model.ModelName](/model/model_name.py) and extend functions
in [model/model_tools.py](/model/model_tools.py) and [aggregation/pipeline_blocks.py](/aggregation/pipeline_blocks.py).

## Pipelines

We take a list of sentences as input, where each sentence is a list of tokens (of type `str`).

1. Lower case all sentences.
2. Wrap each sentence with special tokens `[CLS]` and `[SEP]`.
3. Calculate one contextualized word vector per token with a
   pre-trained [CharacterBERT](https://github.com/helboukkouri/character-bert) model.
4. Collect all corresponding word vectors and references per token.
5. Perform Word Sense Discrimination per token with hierarchical clustering.

Depending on the command line arguments, the clustering uses different criteria to cut the dendrogram. If several
arguments are given, the pipeline performs the highest criterion from the table below. Using the known senses from the
ground truth usually delivers the best results, but ignores tokens without labels. The second to best option is
choosing a maximum distance is most promising, where good initial search ranges are 8-10 for Euclidean distances.

| Argument             | Criterion Description                                                  |
|----------------------|------------------------------------------------------------------------|
| `--known_senses`     | fit the number of senses from the ground truth                         |
| `--max_distance d`   | cuts each dendrogram at a given maximum linkage distance               |
| `--min_silhouette s` | predict the number of senses with the Silhouette Coefficient criterion |

## Evaluation

- [evaluation notebook](/evaluation.ipynb) offers many plots and statistics that deliver insights like sense counts and
  clustering metrics.
- On the one hand, you can browse the generated dictionary in the last section of this notebook as a **DataFrame**. On
  the other hand, you can use the notebook or the `--export_html` pipeline option to generate an **HTML page** with the
  dictionary and corresponding sentences from the training corpus.

## Caches

The software uses caches to enable executions in offline HPC environments and to speed up repeated calculations.

- Models and tokenizers: [model_cache](/model_cache)
- Corpora: [data/corpus_cache](/data/corpus_cache)
- Word vector matrix and raw `id_map` per corpus: user defined result location

## System Requirements

The calculation of word vectors is the only part that benefits from multiple CPU cores. As the first pipeline run caches
the word vectors for reuse, further runs do not need multiple cores. For most corpora and setups, 8GB RAM is sufficient.
We recommend 16-24GB RAM for SemCor and 64GB RAM for Shakespeare.

Pipeline runs with known sense counts only consider tokens that do have sense tags. This optimization reduces run time
and memory footprint during the clustering phase.

## Tests

Run `python -m unittest` in the main directory to execute all tests in [test](/test).
