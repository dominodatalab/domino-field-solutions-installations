
import mlflow
import pandas as pd
loaded_model = mlflow.pyfunc.load_model("/app/model/sklearn-model")

df = pd.DataFrame({'features': [['5', '4.2'], ['3', '7.9'], ['2', '9']],
                       'label': ['1', '0', '0']})
print(loaded_model.predict([[20,0]]))

'''
import mlflow
from pyspark.sql.functions import struct, col
# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, "/app/model/sklearn-model", result_type='double')

# Predict on a Spark DataFrame.
df.withColumn('predictions', loaded_model(struct(*map(col, df.columns))))
'''