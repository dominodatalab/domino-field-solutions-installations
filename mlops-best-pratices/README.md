# MLOPS Examples

## Invoking MLFLOW from an External Agent

Example [code](external_client/run_experiment_and_register_model.py) illustrates how this is done

In the external agent add the following Python [libraries](./requirements.txt) -

1. `dominodatalab==1.2.4`
2. `mlflow==2.9.1` . This needs to be compatible with the appropriate Domino version. Run `pip show mlflow` in a Domino 
workspace to determine the version
   
T
