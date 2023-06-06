# dac-divide-and-conquer

We have implemented a library for extreme multi-label classification that leverages semantic relationship between labels. This library has been extensively tested in disease coding in Spanish and has improved state-of-the-art performance in Codiesp procedures and Codiesp Diagnostics. All the defaults are in the library code.

This code is open and almost self explanatory so please check the code and the arguments. Any doubt can be addressed on the Issues.

For more explanation into the workings of the models, check the following paper: https://aclanthology.org/2022.louhi-1.16.pdf

For citation please use:

```
@inproceedings{barros2022divide,
  title={Divide and Conquer: An Extreme Multi-Label Classification Approach for Coding Diseases and Procedures in Spanish},
  author={Barros, Jose and Rojas, Mat{\'\i}as and Dunstan, Jocelyn and Abeliuk, Andres},
  booktitle={Proceedings of the 13th International Workshop on Health Text Mining and Information Analysis (LOUHI)},
  pages={138--147},
  year={2022}
}
```

## Install library

To use this library run the following command (We recommend creating a Virtual Enviroment in your project first)

```
pip install git+https://github.com/plncmm/dac-divide-and-conquer.git
```

## Reproduce results

To reproduce any of the results you should run the following code changing the path to one you can use for storing intermediate steps of the process:

```
from pathlib import Path
from dac_divide_and_conquer.dataset import CodiespCorpus, CantemistCorpus
from dac_divide_and_conquer.evaluation import eval_mean, eval_ensemble

data_path = Path("..").resolve() / "data"

corpus = CodiespCorpus(data_path, CodiespSubtask.diagnostics)
corpus.download_corpus() # Only works if the method is implemented, else manually download
corpus.create_corpuses() # Preprocess the data to be able to train the model

transformers = (
    ["PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"] * 5
    + ["PlanTL-GOB-ES/roberta-base-biomedical-es"] * 5
    + ["dccuchile/bert-base-spanish-wwm-cased"] * 5
) # List of already tried transformers, 5 each
seeds = list(range(15))
for transformer, seed in zip(transformers, seeds):
    model = DACModel(corpus, matcher_transformer=transformer, seed=seed)
    model.train() # Trains the Ranker and Matcher
    model.save() # Saves the model
    model.eval() # Evals the models and the Ranker and Matcher

eval_mean(corpus, transformers, seeds) # Loads the models and evaluate the mean of them
eval_ensemble(corpus, transformers, seeds) # Loads the models evaluate the ensemble of all
```

You can reproduce the results for the following corpuses: CodiespCorpus-diagnostics, CodiespCorpus-procedures, CantemistCorpus, MESINESPCorpus-abstracts, MESINESPCorpus-clinical_trials. The FALP corpus is a private corpus so please request the data. The same with the Waiting List Corpus. The Clinical Trials corpus is public but its download method is not implemented.

## Use existing models (already trained and saved) to predict new data

```
from pathlib import Path
from dac_divide_and_conquer.dataset import CodiespCorpus
from flair.data import Sentence

data_path = Path("..").resolve() / "data"

corpus = CodiespCorpus(data_path, CodiespSubtask.diagnostics)
model = DACModel.load(corpus, transformer, seed)
sentences = [Sentence("Hola esto quiero predecir"), Sentence("Hola esto tambien")]
model.predict(sentences, return_probabilities=False)
for sentence in sentences:
    predicted = sentence.get_labels("labels_predicted")
    print(f"{sentence} -> {predicted}")
```

## Save models on GCP

If you want to save the models in Google Cloud Storage you need a .env with the following variables

```
GOOGLE_APPLICATION_CREDENTIALS =
PROJECT_ID =
BUCKET =
```

## Transformers that have been used

List of transformers that have been used:

```
 "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
 "PlanTL-GOB-ES/roberta-base-biomedical-es"
 "dccuchile/bert-base-spanish-wwm-cased"
```
