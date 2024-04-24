## Import and Export Models

This section describes how to run example notebooks that makes import and export models to/from a Domino deployment.

### Pre-requisites
Setup a custom environment with the required packages by following the instructions in [financial_sentiment.ipynb](./financial_sentiment.ipynb).

### Importing a Huggingface model
Follow the steps in the [financial_sentiment.ipynb](./financial_sentiment.ipynb) notebook for an example of registering Huggingface models the domino model registry.

### Exporting a registered model
Follow the steps in the [export_model.ipynb](./export_model.ipynb) notebook for an example of downloading the artifacts of a registered model for export to an external location.

NOTE: see the [external_client examples](../external_client/README.md) for instructions on how to run these type of import/export scripts from outside of the domino deployment.