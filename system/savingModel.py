def save_model(model, scaler, model_file="System/model.pkl", scaler_file="System/scaler.pkl"):
    """Save the trained model and scaler to disk."""
    with open(model_file, "wb") as mf:
        pickle.dump(model, mf)
    with open(scaler_file, "wb") as sf:
        pickle.dump(scaler, sf)
