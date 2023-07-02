class TestRegressionSAILPipeline:
    def test_regression_pipeline_partial_fit(
        self, regression_pipeline, regression_dataset
    ):
        X, y = regression_dataset

        batch_size = 50
        for start in range(0, 1001, batch_size):
            end = start + batch_size

            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]

            regression_pipeline.partial_fit(X_train, y_train)

        # Save SAIL pipeline
        regression_pipeline.save(".")

        # Load SAIL pipeline
        new_regression_pipeline = regression_pipeline.load(".")

        for start in range(1001, 2001, batch_size):
            end = start + batch_size

            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]

            new_regression_pipeline.partial_fit(X_train, y_train)

    def test_classification_pipeline_fit(self, regression_pipeline, regression_dataset):
        X, y = regression_dataset

        batch_size = 50
        for start in range(0, 1001, batch_size):
            end = start + batch_size

            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]

            regression_pipeline.fit(X_train, y_train)
