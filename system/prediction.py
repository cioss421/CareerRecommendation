def predict_test_dataset(input_file="test_dataset.csv", model_file="System/model.pkl", scaler_file="System/scaler.pkl", output_file="predicted/prediction.csv"):
    """Predict targets for the test dataset and save only the predictions."""
    try:
        # Load model, scaler, and encoder
        model, scaler = load_model(model_file, scaler_file)

        # Load test dataset
        test_data = pd.read_csv(input_file)
        test_data_cleaned = data_clean_engineering(test_data, fit_encoder=False)

        # Scale the features
        X_test = test_data_cleaned.drop(columns=["Target_Field"], errors="ignore")
        X_test_scaled = scaler.transform(X_test)

        # Make predictions
        predictions = model.predict(X_test_scaled)

        # Save only the last column (Predictions) to the file
        output_df = pd.DataFrame({"Prediction": predictions})
        output_df.to_csv(output_file, index=False)

        print(f"Predictions saved to {output_file}")
        return predictions

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise
