import pandas as pd
import pickle

def preprocess_data(df):

    columns_to_drop = ['SeriousDlqin2yrs', 'Sr.No.']
    df = df.drop(columns=columns_to_drop, axis=1)

    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].mean())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].mode()[0])
    
    return df

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_default_probability(model, data):
    pred_proba = model.predict_proba(data)
    return pred_proba[:, 1]

def main():
    test_data = pd.read_csv("cs-test.csv") 
    test_data = preprocess_data(test_data)
    model = load_model("RandomForestClassifier.pkl")
    default_probabilities = predict_default_probability(model, test_data)
    threshold = 0.5
    
    test_data['SeriousDlqin2yrs'] = (default_probabilities > threshold).astype(int)
    test_data.to_csv("inference_results.csv", index=False)
    print('inference Done.')

if __name__ == "__main__":
    main()
