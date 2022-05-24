# dac-divide-and-conquer

We have implemented a library for extreme multi-label classification that leverages semantic relationship between labels. This library has been extensively tested in disease coding in Spanish and has improved state-of-the-art performance in Codiesp procedures and Codiesp Diagnostics. All the defaults are in the library code.

To use this library run the following command (We recommend creating a Virtual Enviroment in your project first)

```
pip install git+https://github.com/plncmm/dac-divide-and-conquer.git
```

To reproduce any of the results you should run the following code changing the path to one you can use for storing intermediate steps of the process:

```
from pathlib import Path
from dac_divide_and_conquer.dataset import CodiespCorpus, CantemistCorpus, MESINESPCorpus

data_path = Path("..").resolve() / "data"

corpus = CodiespCorpus(data_path, CodiespSubtask.diagnostics)
corpus.reproduce_mean_models()
corpus.reproduce_ensemble_models()
```

You can reproduce the results for the following corpuses: CodiespCorpus-diagnostics, CodiespCorpus-procedures, CantemistCorpus, MESINESPCorpus-abstracts, MESINESPCorpus-clinical_trials.

This code is open and almost self explanatory so please check the code and the arguments. Any doubt can be addressed on the Issues.

If you want to save the models in Google Cloud Storage you need a .env with the following variables

```
GOOGLE_APPLICATION_CREDENTIALS =
PROJECT_ID =
BUCKET =
```

List of transformers that have been used:

```
 "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
 "PlanTL-GOB-ES/roberta-base-biomedical-es"
 "dccuchile/bert-base-spanish-wwm-cased"
```
