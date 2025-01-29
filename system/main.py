if __name__ == "__main__":
   
   train_data = get_data()
   train_data = data_clean_engineering(train_data, fit_encoder=True)
   model, scaler = create_model(train_data)

   if len(sys.argv) != 3:
        print("Usage: python main.py <input_csv> <output_csv>")
        sys.exit(1)
    
   input_csv = sys.argv[1]
   output_csv = sys.argv[2]
   
   # Predict for test dataset
   predict_test_dataset(input_file="data/test_dataset.csv", output_file="predicted/prediction.csv")
