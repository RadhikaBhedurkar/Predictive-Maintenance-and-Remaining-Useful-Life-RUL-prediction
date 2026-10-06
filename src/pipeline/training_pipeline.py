import joblib
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import prepare_train, prepare_test
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import evaluate_and_save
from src.logger import logger

def main():
    logger.info("Training pipeline started")
    print("=== Predictive Maintenance Training Pipeline ===")

    ingestion = DataIngestion()
    train_df = ingestion.load_train()
    test_df = ingestion.load_test()
    rul_df = ingestion.load_rul()

    validator = DataValidation()
    validator.validate(train_df)
    validator.validate(test_df)

    X_train, y_train, feature_columns = prepare_train(train_df)

    trainer = ModelTrainer()
    best, comparison = trainer.train(X_train, y_train)

    print("\nValidation comparison:")
    print(comparison.to_string(index=False))
    print(f"\nSelected model: {best['name']}")

    X_test, y_test, final_test, test_features = prepare_test(test_df, rul_df)

    # Guarantee the same feature order used during training.
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    test_metrics = evaluate_and_save(
        best["model"], X_test, y_test, final_test
    )

    joblib.dump(feature_columns, "artifacts/feature_columns.pkl")
    joblib.dump({"model_name": best["name"]}, "artifacts/model_metadata.pkl")

    print("\nHeld-out FD001 test metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nArtifacts:")
    print("artifacts/model.pkl")
    print("artifacts/feature_columns.pkl")
    print("artifacts/model_metadata.pkl")
    print("artifacts/model_comparison.csv")
    print("artifacts/test_metrics.csv")
    print("artifacts/test_predictions.csv")

    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    main()
