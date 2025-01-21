# Define the Python version neded for CTF
FROM python:3.10-slim

## Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
#
## Keeps Python from buffering stdout and stderr to avoid the framework
## from crashing without emitting any logs due to buffering
ENV PYTHONUNBUFFERED=1

# Label maintainer
LABEL maintainer="Dr. Farhad Allian - The University of Sheffield"

# Create a folder for the source code/outputs
RUN mkdir -p ./causal_testing
RUN mkdir -p ./data/outputs

# Copy the source code from local root and test files from build into the container
COPY --chown=nobody  ./causal_testing ./causal_testing
COPY --chown=nobody ./dafni/src/ ./src
COPY --chown=nobody ./dafni/data/inputs ./data/inputs

# Install core dependencies using PyPi
RUN pip install causal-testing-framework --no-cache-dir

# Set the PYTHONPATH environment variable to include the /src directory
ENV PYTHONPATH="/src:${PYTHONPATH}"

# Define the entrypoint/commands
CMD python -m main_dafni --dag_path $DAG_PATH --data_path $DATA_PATH --tests_path $TESTS_PATH
