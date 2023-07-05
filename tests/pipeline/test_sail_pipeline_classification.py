class TestClassificationSAILPipeline:
    def test_classification_pipeline_partial_fit(
        self, classification_pipeline, classification_dataset
    ):
        X, y = classification_dataset

        batch_size = 50
        for start in range(0, 201, batch_size):
            end = start + batch_size

            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]

            classification_pipeline.partial_fit(
                X_train, y_train, classifier__classes=[1, 0]
            )

        # Save SAIL pipeline
        classification_pipeline.save(".")

        # Load SAIL pipeline
        new_classification_pipeline = classification_pipeline.load(".")

        for start in range(201, 401, batch_size):
            end = start + batch_size

            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]

            new_classification_pipeline.partial_fit(
                X_train, y_train, classifier__classes=[1, 0]
            )

    def test_classification_pipeline_fit(
        self, classification_pipeline, classification_dataset
    ):
        X, y = classification_dataset

        batch_size = 50
        for start in range(0, 201, batch_size):
            end = start + batch_size

            X_train = X.iloc[start:end]
            y_train = y.iloc[start:end]

            classification_pipeline.fit(X_train, y_train, classifier__classes=[1, 0])
