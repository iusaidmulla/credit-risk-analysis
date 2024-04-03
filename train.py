import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE,RandomOverSampler

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_data(self):
        return pd.read_csv(self.file_path)
    
    def preprocess_data(self, df):
        
        print(f'Checking Descriptions: {df.describe(include='all')}')
        print('-' * 50)
        print(f'Dataframe-info: {df.info()}')
        print('-' * 50)
        print(f'Checking Duplicates: {df.duplicated().sum()}')
        print('-' * 50)
        print(f'Checking Null Values:\n{df.isna().sum()}')
        print('-' * 50)
        df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].mean())
        df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].mode()[0])
        
        # Scale the specified columns
        scaler = StandardScaler()
        columns_to_scale = ['NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse']
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        
        X = df.drop(columns=['Sr. No', 'SeriousDlqin2yrs'], axis=1)
        y = df['SeriousDlqin2yrs']
        
        # Balance classes using SMOTE
        # smote = SMOTE(k_neighbors=3,random_state=42)
        # X_resampled, y_resampled = smote.fit_resample(X, y)

        #Random Oversampling
        # random_os = RandomOverSampler(shrinkage=None, sampling_strategy='auto')
        # X_random, y_random = random_os.fit_resample(X, y)

        return X, y
 
class ModelTrainer:
    def __init__(self, model):
        self.model = model
    
    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        y_pred_proba = self.model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba[:,1])
        return roc_auc
    
    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.model, file)

def evaluate_performance(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1


def plot_confusion_matrix(y_true, y_pred, data_type):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - {data_type}')
    plt.savefig(f'{data_type}.png')
    
def main():
    data_processor = DataProcessor("cs-training.csv")
    X, y = data_processor.preprocess_data(data_processor.load_data())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
    # Calculate class weights 
    # class_weights = class_weight.compute_class_weight(class_weight = "balanced",classes = np.unique(y_train),y = y_train) 
    # class_weights_dict = dict(zip(np.unique(y_train), class_weights))

    dTree = DecisionTreeClassifier(criterion='entropy', max_depth=1000, max_features='sqrt', ccp_alpha=0.0001)
    rf = RandomForestClassifier(n_estimators=90,criterion='gini', max_depth=11,min_weight_fraction_leaf=0.0001,max_features='sqrt',
                                class_weight='balanced' ,ccp_alpha=0.0001) 
    model_trainer = ModelTrainer(rf)
    model_trainer.model.fit(X_train, y_train)

    train_accuracy = model_trainer.model.score(X_train, y_train)
    test_accuracy = model_trainer.model.score(X_test, y_test)

    train_roc_auc = model_trainer.evaluate_model(X_train, y_train)
    test_roc_auc = model_trainer.evaluate_model(X_test, y_test)

    y_train_pred = model_trainer.model.predict(X_train)
    y_test_pred = model_trainer.model.predict(X_test)

    # train_precision, train_recall, train_f1 = evaluate_performance(y_train, y_train_pred)
    # test_precision, test_recall, test_f1 = evaluate_performance(y_test, y_test_pred)

    print(f'Train Accuracy: {train_accuracy:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print('-' * 25)
    print(f'Train ROC AUC Score: {train_roc_auc:.4f}')
    print(f'Test ROC AUC Score: {test_roc_auc:.4f}')
    # print('-' * 25)
    # print(f"Train Precision: {train_precision:.4f}")
    # print(f"Test Precision: {test_precision:.4f}")
    # print('-' * 25)
    # print(f"Train Recall: {train_recall:.4f}")
    # print(f"Test Recall: {test_recall:.4f}")
    # print('-' * 25)
    # print(f"Train F1-Score: {train_f1:.4f}")
    # print(f"Test F1-Score: {test_f1:.4f}")

    plot_confusion_matrix(y_train, y_train_pred, "Train-Data")
    plot_confusion_matrix(y_test, y_test_pred, "Test-Data")
    model_trainer.save_model("RandomForestClassifier.pkl")

if __name__ == "__main__":
    main()