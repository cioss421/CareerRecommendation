

import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

def get_data():
    train_df = pd.read_csv("data/train_dataset.csv")
    return train_df

def data_clean_engineering(train_data, encoder=None, fit_encoder=False):

    column_to_move = "Target_Field"
    target_col = train_data.pop(column_to_move)
    train_data[column_to_move] = target_col
    
    #Feature Engineering: Creating a new feature based on Tech_Savvy_Score, Creativity_Score, Study_Hours, Age
    train_data['Field_Fit_Score'] = (
        0.4 * train_data['Tech_Savvy_Score'] +
        0.3 * train_data['Creativity_Score'] +
        0.2 * train_data['Study_Hours'] +
        0.1 * (train_data['Age'] / max(train_data['Age']))
        )

    #Feature Engineering: Creating a score tab based on what primary skill they have.
    train_data['Primary_Skill_Score'] = train_data['Primary_Skill'].replace({'Adaptability' : 5, 
                                                             'Technical Skills' : 10, 
                                                             'Creativity' : 8, 
                                                             'Teamwork' : 9, 
                                                             'Leadership' : 8, 
                                                             'Communication' : 6, 
                                                             'Problem-Solving' : 10, 
                                                             "Critical Thinking" : 10})
    
    #Feature Engineering: Creating a new features based on Tech_Savvy_Score and Primary_Skill_Score
    train_data['Skill_Utilization'] = 0.5 * train_data['Tech_Savvy_Score'] + 0.5 * train_data['Primary_Skill_Score']

    train_data['Academic_Readiness'] = train_data['Tech_Savvy_Score'] + train_data['Creativity_Score'] + train_data['Study_Hours'] / 3

    #Data Engineering: Replacing categorical values by numerical values
    train_data['Family_Income_Bracket'].replace({"High": 3, "Middle": 2, "Low": 1}, inplace=True)
    train_data['Scholarship_Status'] = train_data['Scholarship_Status'].apply(lambda x: 1 if x == "Yes" else 0)
    train_data['Cultural_Influence'].replace({"Society": 3, "Personal Interest": 2, "Family": 1}, inplace=True)

    #Feature Engineering: Getting the SocioEconomic Score base on Family_Income_Bracket, Scholarship_Status, Cultural_Influence
    train_data['SocioEconmic'] = 3 / ((1/train_data['Family_Income_Bracket']) + (1/train_data['Scholarship_Status']) + (1/train_data['Cultural_Influence']))

    #Feature Engineering: alignment of feature to target
    train_data['Strand_Target_Matching'] = train_data.apply(check_alignment, axis=1)

    #Data Cleaning: dropping unnecessary feauters
    train_data.drop(['Primary_Skill', 'Hobby'], axis=1, inplace=True)

    #Data Engineering: converting categorical values into numerical/nominal (0,1,2...n)
    selected_col = train_data[['MBTI','Extracurricular', 'Target_Field']]
    for col in selected_col.columns:
        track_mapping = {track: idx for idx, track in enumerate(train_data[col].unique())}
        train_data[col] = train_data[col].map(track_mapping)
    
    #Data Engineering: converting categorical features into numerical type
    cols_encode = ['Strand', 'Personality_Type', 'Future_Field_Security', 'Work_Flexibility']
    if fit_encoder:
        encoder = OneHotEncoder(sparse_output=False, drop=None)
        encoded = encoder.fit_transform(train_data[cols_encode])
        with open("System/encoder.pkl", "wb") as ef:
            pickle.dump(encoder, ef)
    else:
        with open("System/encoder.pkl", "rb") as ef:
            encoder = pickle.load(ef)
        encoded = encoder.transform(train_data[cols_encode])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
    train_data = pd.concat([train_data, encoded_df], axis=1)  
    train_data.drop(columns=cols_encode, inplace=True)

    #Data engineering: converting categorical feature into nominal values.
    train_data['Scholarship_Status'].replace({'Yes': 1, 'No': 0}, inplace=True)
    train_data['Family_Income_Bracket'].replace({'Low': 0, 'Middle': 1, 'High': 2}, inplace=True)
    train_data['Cultural_Influence'].replace({'Family': 0, 'Personal Interest': 1, 'Society': 2}, inplace=True)

    cat_columns = ['Family_Income_Bracket', 'Cultural_Influence', 'Scholarship_Status']
    train_data = cat_correlation(train_data, cat_columns)

    return train_data

# Feature Engineering: checking the alignment of strand to the course/field.
def check_alignment(row):

    strand = row['Strand']
    target_field = row['Target_Field']

    strand_target_mapping = {
        "STEM": ["Healthcare", "Engineering & Architecture", "Science and Research", "Energy and Utilities"],
        "ABM": ["Business/Commerce", "Hospitality and Tourism", "Public Service"],
        "HUMSS": ["Public Service", "Law and Legal Services", "Education", "Media and Communication"],
        "ICT": ["Engineering & Architecture", "Media and Communication", "Science and Research"],
        "TVL": ["Trades and Crafts", "Hospitality and Tourism", "Logistics and Transportation"],
        "GAS": ["Any"],
    }

    if strand == "GAS":
        return 1
        
    return 1 if target_field in strand_target_mapping.get(strand, []) else 0

# Feature Engineering: Composition Correlation of 'Family_Income_Bracket', 'Cultural_Influence', 'Scholarship_Status'
def cat_correlation(df, cat_columns):

    correlation_values = []
    correlation_matrix = df.corr()
    for col in cat_columns:
        cat_corr = correlation_matrix[col].drop(cat_columns)
        weighted_corr = np.abs(cat_corr).sum()
        correlation_values.append(weighted_corr)

    normalized_correlations = np.array(correlation_values) / np.sum(correlation_values)

    df['Composite_Correlation'] = np.dot(df[cat_columns], normalized_correlations)

    return df

# Training the model with train data
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

# Saving the model and scaler using pickle
def save_model(model, scaler, model_file="System/model.pkl", scaler_file="System/scaler.pkl"):
    """Save the trained model and scaler to disk."""
    with open(model_file, "wb") as mf:
        pickle.dump(model, mf)
    with open(scaler_file, "wb") as sf:
        pickle.dump(scaler, sf)

# Retrieve the model and scaler using pickle
def load_model(model_file="System/model.pkl", scaler_file="System.scaler.pkl"):
    """Load the model and scaler from disk."""
    with open(model_file, "rb") as mf:
        model = pickle.load(mf)
    with open(scaler_file, "rb") as sf:
        scaler = pickle.load(sf)
    return model, scaler

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
