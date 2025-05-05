import pandas as pd
import numpy as np
import glob
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


class PhishingSpamDetector:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf = None
        self.rf_model = None
        self.xgb_model = None
        self.label_encoder = LabelEncoder()

    def load_and_combine_data(self):
        """Load and combine all CSV files in the directory"""
        all_files = glob.glob("*.csv")
        df_list = []
        
        for f in all_files:
            try:
                # Try multiple encodings if default fails
                try:
                    temp_df = pd.read_csv(f)
                except UnicodeDecodeError:
                    temp_df = pd.read_csv(f, encoding='latin1')
                df_list.append(temp_df)
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
                try:
                    temp_df = pd.read_csv(f, sep=None, engine='python', on_bad_lines='skip')
                    df_list.append(temp_df)
                except Exception as e:
                    print(f"Failed to load {f}: {str(e)}")
                    continue
        
        if not df_list:
            raise ValueError("No CSV files were successfully loaded.")
        
        return pd.concat(df_list, ignore_index=True)

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

    def train_and_evaluate(self):
        """Train and evaluate models using k-fold cross-validation"""
        # Step 1: Load and combine data
        print("Loading and combining data...")
        combined_df = self.load_and_combine_data()
        print(f"Combined data shape: {combined_df.shape}")
        
        # Step 2: Identify text and label columns
        text_col = next((col for col in ['text', 'body', 'message', 'content', 'email'] if col in combined_df.columns), None)
        label_col = next((col for col in ['label', 'spam', 'type', 'class', 'target'] if col in combined_df.columns), None)
        
        if text_col is None or label_col is None:
            print("Could not find text or label columns. Available columns:")
            print(combined_df.columns.tolist())
            return
        
        print(f"Using '{text_col}' as text column and '{label_col}' as label column")
        
        # Step 3: Preprocess labels - convert to three categories
        # First, standardize the labels
        combined_df['label_standardized'] = combined_df[label_col].str.lower().str.strip()
        
        # Map to our three categories
        label_mapping = {
            'ham': 'ham',
            'legitimate': 'ham',
            'safe': 'ham',
            'real': 'ham',
            '0': 'ham',
            'spam': 'spam',
            '1': 'spam',
            'phishing': 'phishing',
            'phish': 'phishing',
            '2': 'phishing'
        }
        
        combined_df['label_category'] = combined_df['label_standardized'].map(label_mapping)
        
        # For any unmapped labels, we'll classify them based on content
        unmapped = combined_df['label_category'].isna()
        if unmapped.any():
            print(f"Found {unmapped.sum()} unmapped labels. Attempting to classify them...")
            # Simple heuristic - if contains http or www, likely phishing
            combined_df.loc[unmapped & combined_df[text_col].str.contains(r'http|www', case=False, regex=True), 'label_category'] = 'phishing'
            # Otherwise mark as spam (since original code treated everything as binary spam/phishing)
            combined_df.loc[unmapped & combined_df['label_category'].isna(), 'label_category'] = 'spam'
        
        # Encode the labels numerically
        self.label_encoder.fit(combined_df['label_category'])
        combined_df['label_encoded'] = self.label_encoder.transform(combined_df['label_category'])
        
        # Step 4: Preprocess text
        print("Preprocessing text...")
        combined_df['processed_text'] = combined_df[text_col].apply(self.preprocess_text)
        
        # Step 5: TF-IDF Vectorization
        print("Creating TF-IDF features...")
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self.tfidf.fit_transform(combined_df['processed_text'])
        y = combined_df['label_encoded'].values
        
        # Step 6: Train and evaluate models
        print("\nTraining models...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        self.rf_model = RandomForestClassifier(random_state=42, n_estimators=200, class_weight='balanced')
        self.xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss', n_estimators=200)
        
        rf_accuracies, xgb_accuracies = [], []
        rf_cm, xgb_cm = np.zeros((3, 3)), np.zeros((3, 3))  # For combined confusion matrices
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n=== Fold {fold + 1} ===")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train Random Forest
            self.rf_model.fit(X_train, y_train)
            rf_pred = self.rf_model.predict(X_val)
            rf_acc = accuracy_score(y_val, rf_pred)
            rf_accuracies.append(rf_acc)
            rf_cm += confusion_matrix(y_val, rf_pred)
            print(f"Random Forest Accuracy: {rf_acc:.4f}")
            print(classification_report(y_val, rf_pred, target_names=self.label_encoder.classes_, zero_division=0))
            
            # Train XGBoost
            self.xgb_model.fit(X_train, y_train)
            xgb_pred = self.xgb_model.predict(X_val)
            xgb_acc = accuracy_score(y_val, xgb_pred)
            xgb_accuracies.append(xgb_acc)
            xgb_cm += confusion_matrix(y_val, xgb_pred)
            print(f"XGBoost Accuracy: {xgb_acc:.4f}")
            print(classification_report(y_val, xgb_pred, target_names=self.label_encoder.classes_, zero_division=0))
        
        # Plot confusion matrices
        self.plot_confusion_matrix(rf_cm, "Random Forest")
        self.plot_confusion_matrix(xgb_cm, "XGBoost")
        
        print("\n=== Final Results ===")
        print(f"Random Forest Mean Accuracy: {np.mean(rf_accuracies):.4f}")
        print(f"XGBoost Mean Accuracy: {np.mean(xgb_accuracies):.4f}")
        
        # Step 7: Save models
        self.save_models()

    def plot_confusion_matrix(self, cm, model_name):
        """Plot normalized confusion matrix"""
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_, 
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def save_models(self):
        """Save trained models and vectorizer"""
        print("\nSaving models...")
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(self.tfidf, "saved_models/tfidf_vectorizer.joblib")
        joblib.dump(self.rf_model, "saved_models/random_forest.joblib")
        joblib.dump(self.xgb_model, "saved_models/xgboost.joblib")
        joblib.dump(self.label_encoder, "saved_models/label_encoder.joblib")
        print("Models saved to 'saved_models' directory")

    @staticmethod
    def load_models():
        """Load pre-trained models"""
        detector = PhishingSpamDetector()
        try:
            detector.tfidf = joblib.load("saved_models/tfidf_vectorizer.joblib")
            detector.rf_model = joblib.load("saved_models/random_forest.joblib")
            detector.xgb_model = joblib.load("saved_models/xgboost.joblib")
            detector.label_encoder = joblib.load("saved_models/label_encoder.joblib")
            print("Models loaded successfully!")
            return detector
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")

    def predict(self, emails, model='both'):
        """
        Make predictions on new emails
        
        Args:
            emails: Single email text or list of emails
            model: 'rf' for Random Forest, 'xgb' for XGBoost, 'both' for both
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if isinstance(emails, str):
            emails = [emails]
            
        processed_emails = [self.preprocess_text(email) for email in emails]
        X = self.tfidf.transform(processed_emails)
        
        results = {'emails': emails}
        
        if model in ['rf', 'both']:
            rf_pred = self.rf_model.predict(X)
            rf_proba = self.rf_model.predict_proba(X)
            results['random_forest'] = {
                'predictions': self.label_encoder.inverse_transform(rf_pred).tolist(),
                'probabilities': rf_proba.tolist(),
                'class_labels': self.label_encoder.classes_.tolist()
            }
            
        if model in ['xgb', 'both']:
            xgb_pred = self.xgb_model.predict(X)
            xgb_proba = self.xgb_model.predict_proba(X)
            results['xgboost'] = {
                'predictions': self.label_encoder.inverse_transform(xgb_pred).tolist(),
                'probabilities': xgb_proba.tolist(),
                'class_labels': self.label_encoder.classes_.tolist()
            }
            
        return results

    def print_predictions(self, prediction_results):
        """Print prediction results in readable format"""
        for i, email in enumerate(prediction_results['emails']):
            print(f"\nEmail {i+1}:")
            print(f"Text: {email[:100]}..." if len(email) > 100 else f"Text: {email}")
            
            if 'random_forest' in prediction_results:
                rf_pred = prediction_results['random_forest']['predictions'][i]
                rf_proba = prediction_results['random_forest']['probabilities'][i]
                rf_labels = prediction_results['random_forest']['class_labels']
                max_idx = np.argmax(rf_proba)
                print(f"Random Forest Prediction: {rf_pred.upper()}")
                print("Confidence Scores:")
                for label, prob in zip(rf_labels, rf_proba):
                    print(f"  {label.upper()}: {prob:.2f}")
                
            if 'xgboost' in prediction_results:
                xgb_pred = prediction_results['xgboost']['predictions'][i]
                xgb_proba = prediction_results['xgboost']['probabilities'][i]
                xgb_labels = prediction_results['xgboost']['class_labels']
                max_idx = np.argmax(xgb_proba)
                print(f"\nXGBoost Prediction: {xgb_pred.upper()}")
                print("Confidence Scores:")
                for label, prob in zip(xgb_labels, xgb_proba):
                    print(f"  {label.upper()}: {prob:.2f}")
            print("\n" + "="*50)

def main():
    # Initialize detector
    detector = PhishingSpamDetector()
    
    # Train new models (comment out if you just want to use pre-trained)
    detector.train_and_evaluate()
    
    # Or load pre-trained models
    # detector = PhishingSpamDetector.load_models()
    
    # Example usage
    test_emails = [
        "Hi there, just checking in about our meeting tomorrow at 2pm.",
        "Congratulations! You've won a $1000 Amazon gift card. Click here to claim: bit.ly/fakelink",
        "URGENT: Your bank account has been compromised. Please verify your details immediately: http://phishybank.com/login",
        "The quarterly report is attached for your review. Best regards, Accounting Team",
        "Limited time offer! Buy now and get 50% off on all products. Visit our store today!",
        "Dear customer, your account has been locked. Please login to unlock: http://fakebanklogin.com"
    ]
    
    # Get predictions
    predictions = detector.predict(test_emails)
    
    # Print results
    detector.print_predictions(predictions)
    
    # Example of saving predictions to CSV
    results = []
    for i, email in enumerate(test_emails):
        entry = {
            'email': email,
            'rf_prediction': predictions['random_forest']['predictions'][i],
            'xgb_prediction': predictions['xgboost']['predictions'][i]
        }
        
        # Add probabilities for each class
        for j, label in enumerate(predictions['random_forest']['class_labels']):
            entry[f'rf_prob_{label}'] = predictions['random_forest']['probabilities'][i][j]
        
        for j, label in enumerate(predictions['xgboost']['class_labels']):
            entry[f'xgb_prob_{label}'] = predictions['xgboost']['probabilities'][i][j]
        
        results.append(entry)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('email_classification_results.csv', index=False)
    print("\nPredictions saved to 'email_classification_results.csv'")

if __name__ == "__main__":
    main()
