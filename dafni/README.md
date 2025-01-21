# Causal Testing Framework on DAFNI

- This directory contains the containerisation files of the causal testing framework using Docker, which is used
to upload the framework onto [DAFNI](https://www.dafni.ac.uk).
- It is **not** recommended to install the causal testing framework using Docker, and should only be installed
  using [PyPI](https://pypi.org/project/causal-testing-framework/).

### Folders

- `data` contains two sub-folders (the structure is important for DAFNI).
  - `inputs` is a folder that contains the input files that are (separately) uploaded to DAFNI.
    - `causal_tests.json` is a JSON file that contains the causal tests.
    - `dag.dot` is a dot file that contains the directed acyclic graph (dag). In this file, Causal Variables are defined as 
       node metadata attributes as key-value pairs using the following syntax: 
       `node [datatype="int", typestring="input"]`. The `datatype` key specifies the datatype of the causal variable
       as a string (e.g. `"int"`, `"str"`) and the `typestring` key specifies its typestring, which is also a string 
       representing the variable type (e.g. `"input"` or `"output"`).
    - `runtime_data.csv` is a csv file that contains the runtime data.

  - `outputs` is a folder where the `causal_tests_results.json` output file is created.

### Docker files
- `main_dafni.py` is the entry-point to the causal testing framework that is used by Docker.
- `model_definition.yaml` is the model metadata that is required to be uploaded to DAFNI.
- `.env` is an example of a configuration file containing the environment variables. This is only required
    if using `docker-compose` to build the image. 
- `Dockerfile` is the main blueprint that builds the image.
- `.dockerignore` tells the Dockerfile which files to not include in the image.
- `docker-compose.yaml` is another method of building the image and running the container in one line. 
   Note: the `.env` file that contains the environment variables for `main_dafni.py` is only used here.


