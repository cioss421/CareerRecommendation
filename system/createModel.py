def create_model(train_data):
    X_train = train_data.drop('Target_Field', axis=1)
    y_train = train_data['Target_Field']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)

    save_model(model, scaler)
    print("Model, scaler, and encoder saved successfully!")

    return model, scaler
