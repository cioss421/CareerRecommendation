def load_model(model_file="System/model.pkl", scaler_file="System.scaler.pkl"):
    """Load the model and scaler from disk."""
    with open(model_file, "rb") as mf:
        model = pickle.load(mf)
    with open(scaler_file, "rb") as sf:
        scaler = pickle.load(sf)
    return model, scaler
