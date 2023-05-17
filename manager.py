import mlflow


class MLflowModelManager:
    def __init__(self,
                 model,
                 model_name,
                 tracking_uri,
                 run_id) -> None:
        self.model = model
        self.model_name = model_name
        self.run_id = run_id
        self.tracking_uri = tracking_uri
        self.run_uri = self.__create_uri()
        self.model_version = None
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)

    def __create_uri(self):
        return f'runs:/{self.run_id}'

    def register_model(self) -> None:
        self.model_version = mlflow.register_model(self.run_uri, self.model_name)

    def promote_stage(self, origin_stage, destination_stage) -> None:
        """
        Accepted Stages, None, Staging, Production
        """
        if not self.model_version:
            self.model_version = client.get_latest_versions(model_name,
                                                            stages=[origin_stage])[0]
        self.client.transition_model_version_stage(name=self.model_name,
                                                   version=self.model_version.version,
                                                   stage=destination_stage)
    # registered_model = client.get_registered_model('model_from_lecture_03_mlops')
    # version=registered_model.latest_versions[-1]
    # model = mlflow.pyfunc.load_model(version.source)
