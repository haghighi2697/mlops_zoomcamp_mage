import mlflow
import mlflow.sklearn
import pickle
 
if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_model(data):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    dv, lr = data

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("taxi_duration_prediction")
    

    # Start a new MLflow run
    with mlflow.start_run():
        # Log the model
        mlflow.sklearn.log_model(lr, "linear_regression_model")
        print("Model logged successfully.")
        
        # Save and log the dict vectorizer as an artifact
        artifact_path = "dict_vectorizer.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact(artifact_path, "artifacts")
        print("Artifact logged successfully.")
        
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("intercept", lr.intercept_)
        print("Parameters and metrics logged successfully.")