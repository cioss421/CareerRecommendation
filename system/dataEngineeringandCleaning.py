def data_clean_engineering(train_data, encoder=None, fit_encoder=False):

    column_to_move = "Target_Field"
    target_col = train_data.pop(column_to_move)
    train_data[column_to_move] = target_col
    
    train_data['Field_Fit_Score'] = (
        0.4 * train_data['Tech_Savvy_Score'] +
        0.3 * train_data['Creativity_Score'] +
        0.2 * train_data['Study_Hours'] +
        0.1 * (train_data['Age'] / max(train_data['Age']))
        )

    train_data['Primary_Skill_Score'] = train_data['Primary_Skill'].replace({'Adaptability' : 5, 
                                                             'Technical Skills' : 10, 
                                                             'Creativity' : 8, 
                                                             'Teamwork' : 9, 
                                                             'Leadership' : 8, 
                                                             'Communication' : 6, 
                                                             'Problem-Solving' : 10, 
                                                             "Critical Thinking" : 10})
    
    train_data['Skill_Utilization'] = 0.5 * train_data['Tech_Savvy_Score'] + 0.5 * train_data['Primary_Skill_Score']
    train_data['Academic_Readiness'] = train_data['Tech_Savvy_Score'] + train_data['Creativity_Score'] + train_data['Study_Hours'] / 3

    train_data['Family_Income_Bracket'].replace({"High": 3, "Middle": 2, "Low": 1}, inplace=True)
    train_data['Scholarship_Status'] = train_data['Scholarship_Status'].apply(lambda x: 1 if x == "Yes" else 0)
    train_data['Cultural_Influence'].replace({"Society": 3, "Personal Interest": 2, "Family": 1}, inplace=True)

    train_data['SocioEconmic'] = 3 / ((1/train_data['Family_Income_Bracket']) + (1/train_data['Scholarship_Status']) + (1/train_data['Cultural_Influence']))

    train_data['Strand_Target_Matching'] = train_data.apply(check_alignment, axis=1)

    train_data.drop(['Primary_Skill', 'Hobby'], axis=1, inplace=True)

    selected_col = train_data[['MBTI','Extracurricular', 'Target_Field']]
    for col in selected_col.columns:
        track_mapping = {track: idx for idx, track in enumerate(train_data[col].unique())}
        train_data[col] = train_data[col].map(track_mapping)
    
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

    train_data['Scholarship_Status'].replace({'Yes': 1, 'No': 0}, inplace=True)
    train_data['Family_Income_Bracket'].replace({'Low': 0, 'Middle': 1, 'High': 2}, inplace=True)
    train_data['Cultural_Influence'].replace({'Family': 0, 'Personal Interest': 1, 'Society': 2}, inplace=True)

    cat_columns = ['Family_Income_Bracket', 'Cultural_Influence', 'Scholarship_Status']
    train_data = cat_correlation(train_data, cat_columns)

    return train_data

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
