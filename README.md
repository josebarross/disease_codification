# disease-codfication

This repository holds the code to reproduce and train an Information Retrieval model for ICD 10 Code Text Classification

To install this as a library you should create a virtual enviroment and install the library with this command:

```
pip install git+https://your-github-personal-token@github.com/plncmm/disease_codification.git
```

To reproduce results you should run the following code changing the path to one you can use for storing intermediate steps of the process:

```
from pathlib import Path
from disease_codification.reproducibility.codiesp import reproduce_model_codiesp
reproduce_model_codiesp(Path(".").resolve())
```

If you want to save the models you need a .env with the following variables for saving them in a google cloud storage.

```
GOOGLE_APPLICATION_CREDENTIALS =
PROJECT_ID =
BUCKET =
```

All the defaults are in the library code.
