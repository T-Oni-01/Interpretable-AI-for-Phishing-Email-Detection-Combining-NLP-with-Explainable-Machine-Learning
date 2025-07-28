import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import eli5
import joblib
from lime import lime_text
import warnings
import logging
from tqdm import tqdm
from bs4 import BeautifulSoup
import html
import os
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox, filedialog
import webbrowser
from PIL import Image, ImageTk
import base64
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Configuration
EXPLANATION_SAMPLE_SIZE = 50
LIME_SAMPLES = 3
RANDOM_STATE = 42
MAX_FEATURES = 2000  # Reduced features for stability


class PhishingDetector:
    def __init__(self):
        self.initialize_nltk()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))  # Add bigrams
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=50,  # Reduce for faster training
                max_depth=15,
                n_jobs=-1  # Use all CPU cores
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=5,
                tree_method='hist',  # Faster training
                n_jobs=-1
            )
        }
        self.current_model_name = 'Random Forest'  # Default model
        self.model = self.models[self.current_model_name]
        self.is_trained = False
        self.feature_names = None

    def set_model(self, model_name):
        """Switch between different models"""
        if model_name in self.models:
            self.current_model_name = model_name
            self.model = self.models[model_name]
            logger.info(f"Switched to {model_name} model")
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def initialize_nltk(self):
        """Ensure all required NLTK resources are downloaded"""
        resources = {
            'corpora/stopwords': 'stopwords',
            'tokenizers/punkt': 'punkt',
            'corpora/wordnet': 'wordnet'
        }
        for path, package in resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                logger.info(f"Downloading NLTK {package}...")
                nltk.download(package, quiet=True)

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens
                  if word not in self.stop_words and len(word) > 2]
        return ' '.join(tokens)

    def load_data(self, filepath):
        """Load and preprocess data with progress tracking"""
        logger.info("Loading data...")

        # Read CSV with explicit handling of missing values
        df = pd.read_csv(filepath, na_values=['', ' ', 'NaN', 'N/A', 'NA', 'null'])

        # Fill missing values
        df['body'] = df['body'].fillna('')
        df['subject'] = df['subject'].fillna('')

        # Verify label column exists
        if 'label' not in df.columns:
            raise ValueError("Dataset must contain 'label' column")

        # Clean label column - remove any whitespace and convert to string
        df['label'] = df['label'].astype(str).str.strip()

        # Keep only rows where label is '0' or '1'
        valid_labels = df['label'].isin(['0', '1'])
        invalid_count = len(df) - valid_labels.sum()

        if invalid_count > 0:
            logger.warning(f"Dropping {invalid_count} rows with invalid labels")
            df = df[valid_labels].copy()

        # Convert to integers now that we have only valid values
        df['label'] = df['label'].astype(int)

        # Check for NaN in labels (shouldn't happen after above filtering)
        if df['label'].isna().any():
            logger.warning("Found NaN values in labels - dropping these rows")
            df = df.dropna(subset=['label'])

        logger.info(f"Loaded {len(df)} emails after cleaning")

        # Combine subject and body for better analysis
        df['full_text'] = df['subject'] + ' ' + df['body']

        logger.info("Preprocessing text...")
        tqdm.pandas(desc="Preprocessing")
        df['cleaned_text'] = df['full_text'].progress_apply(self.preprocess_text)

        return df

    def train(self, X_train, y_train):
        """Train model with feature importance tracking"""
        # Validate inputs
        if X_train.shape[0] != len(y_train):
            raise ValueError("X_train and y_train have different number of samples")

        if X_train.shape[0] == 0:
            raise ValueError("No training samples provided")

        logger.info(f"Training {self.current_model_name} model with {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Save feature importance visualization
        if hasattr(self.model, 'feature_importances_'):
            self.save_feature_importance_plot()

        return self.model

    def save_feature_importance_plot(self):
        """Save feature importance plot"""
        top_n = 20
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Important Features')
        plt.barh(range(top_n), importances[indices], align='center')
        plt.yticks(range(top_n), [self.feature_names[i] for i in indices])
        plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Feature importance plot saved")

    def predict_email(self, subject, body, return_explanation=True):
        """Predict if an email is phishing with explanation"""
        if not self.is_trained:
            raise Exception("Model not trained. Please train or load a trained model first.")

        # Combine subject and body
        full_text = f"{subject} {body}"
        cleaned_text = self.preprocess_text(full_text)

        # Transform text to features
        text_features = self.tfidf.transform([cleaned_text])

        # Make prediction
        prediction = self.model.predict(text_features)[0]
        proba = self.model.predict_proba(text_features)[0]

        if not return_explanation:
            return prediction, proba

        # Generate explanations
        explanation = self.generate_explanation(cleaned_text, full_text, text_features, prediction)

        return prediction, proba, explanation

    def generate_explanation(self, cleaned_text, original_text, text_features, prediction):
        """Generate multiple types of explanations"""
        explanation = {
            'eli5': None,
            'lime': None,
            'features': None
        }

        # Get top contributing features
        explanation['features'] = self.get_top_features(text_features, prediction)

        # Generate ELI5 explanation
        try:
            html_explanation = eli5.format_as_html(
                eli5.explain_prediction(
                    self.model,
                    text_features[0],
                    feature_names=self.feature_names,
                    target_names=['Legitimate', 'Phishing']
                )
            )
            explanation['eli5'] = html_explanation
        except Exception as e:
            logger.error(f"ELI5 failed: {str(e)}")

        # Generate LIME explanation
        try:
            explainer_lime = lime_text.LimeTextExplainer(
                class_names=['Legitimate', 'Phishing'],
                kernel_width=25
            )

            def predict_fn(texts):
                return self.model.predict_proba(self.tfidf.transform(texts))

            exp = explainer_lime.explain_instance(
                cleaned_text,
                predict_fn,
                num_features=10,
                top_labels=1
            )

            # Convert to HTML
            lime_html = exp.as_html()
            explanation['lime'] = lime_html
        except Exception as e:
            logger.error(f"LIME failed: {str(e)}")

        return explanation

    def get_top_features(self, text_features, prediction, top_n=10):
        """Get top contributing features for a prediction"""
        # Get feature coefficients (for random forest, we use feature importances)
        if hasattr(self.model, 'feature_importances_'):
            coefficients = self.model.feature_importances_
        else:
            coefficients = np.ones(text_features.shape[1])

        # Get non-zero features
        feature_array = text_features.toarray()[0]
        nonzero_indices = feature_array.nonzero()[0]

        # Calculate contribution scores
        contributions = feature_array[nonzero_indices] * coefficients[nonzero_indices]

        # Get top features for the predicted class
        if prediction == 1:  # Phishing
            top_indices = np.argsort(-contributions)[:top_n]
        else:  # Legitimate
            top_indices = np.argsort(contributions)[:top_n]

        top_features = []
        for idx in top_indices:
            feature_idx = nonzero_indices[idx]
            feature_name = self.feature_names[feature_idx]
            value = feature_array[feature_idx]
            contribution = contributions[idx]
            top_features.append({
                'feature': feature_name,
                'value': value,
                'contribution': contribution
            })

        return top_features

    def save_model(self, directory="model"):
        """Save the trained model and vectorizer"""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, os.path.join(directory, 'phishing_model.pkl'))
        joblib.dump(self.tfidf, os.path.join(directory, 'tfidf_vectorizer.pkl'))

        # Save feature names
        with open(os.path.join(directory, 'feature_names.txt'), 'w', encoding='utf-8') as f:
            for name in self.feature_names:
                f.write(f"{name}\n")

        logger.info(f"Model saved to {directory}")

    def load_model(self, directory="model"):
        """Load a trained model and vectorizer"""
        self.model = joblib.load(os.path.join(directory, 'phishing_model.pkl'))
        self.tfidf = joblib.load(os.path.join(directory, 'tfidf_vectorizer.pkl'))

        # Load feature names
        with open(os.path.join(directory, 'feature_names.txt'), 'r', encoding='utf-8') as f:
            self.feature_names = [line.strip() for line in f.readlines()]

        self.is_trained = True
        logger.info(f"Model loaded from {directory}")

    def train_from_csv(self, csv_path):
        """Train the model from a CSV file"""
        df = self.load_data(csv_path)

        # Feature extraction
        X = self.tfidf.fit_transform(df['cleaned_text'])
        y = df['label']
        self.feature_names = self.tfidf.get_feature_names_out()

        # Additional validation
        if X.shape[0] == 0:
            raise ValueError("No valid training samples after preprocessing")

        # Train-test split (for evaluation)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        # Train model
        self.train(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Save evaluation metrics
        self.save_evaluation_metrics(X_test, y_test, y_pred)

        return y_test, y_pred, report

    def save_evaluation_metrics(self, X_test, y_test, y_pred):
        """Save evaluation metrics and visualizations"""
        # Classification report
        report = classification_report(y_test, y_pred)
        with open('classification_report.txt', 'w') as f:
            f.write(report)

        # Confusion matrix
        self.generate_confusion_matrix(y_test, y_pred)

        # ROC curve
        self.plot_roc_curve(X_test, y_test)

    def generate_confusion_matrix(self, y_true, y_pred):
        """Generate confusion matrices"""
        cm = confusion_matrix(y_true, y_pred)

        # Raw counts
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legitimate', 'Phishing'],
                    yticklabels=['Legitimate', 'Phishing'])
        plt.title('Confusion Matrix (Counts)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix_counts.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Normalized
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=['Legitimate', 'Phishing'],
                    yticklabels=['Legitimate', 'Phishing'])
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix_normalized.png', bbox_inches='tight', dpi=300)
        plt.close()

        logger.info("Confusion matrices saved")

    def plot_roc_curve(self, X_test, y_test):
        """Generate ROC curve"""
        y_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png', bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"ROC curve saved (AUC = {roc_auc:.2f})")


class PhishingDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Phishing Email Detector")
        self.root.geometry("900x700")

        # Create GUI elements first
        self.create_widgets()  # This creates all tabs including detection_tab

        # Initialize detector after GUI is set up
        self.detector = PhishingDetector()

        # Now create model selection (after detection_tab exists)
        self.create_model_selection()

        # Load default model if exists
        if os.path.exists("model"):
            try:
                self.detector.load_model()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

        # Load logo image
        self.load_logo()

    def create_widgets(self):
        """Create all GUI widgets"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        # Create tabs
        self.detection_tab = ttk.Frame(self.notebook)  # Store as instance variable
        self.notebook.add(self.detection_tab, text="Detect Phishing")

        self.create_detection_tab_content()  # Separate method for tab content
        self.create_training_tab()
        self.create_about_tab()

    def create_detection_tab_content(self):
        """Create content for the detection tab"""
        # Email input frame
        input_frame = ttk.LabelFrame(self.detection_tab, text="Email Details", padding=10)
        input_frame.pack(fill='x', padx=10, pady=5)

        # Subject
        ttk.Label(input_frame, text="Subject:").grid(row=0, column=0, sticky='w')
        self.subject_entry = ttk.Entry(input_frame, width=80)
        self.subject_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        # Body
        ttk.Label(input_frame, text="Body:").grid(row=1, column=0, sticky='nw')
        self.body_text = scrolledtext.ScrolledText(input_frame, width=80, height=15)
        self.body_text.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')

        # From address (optional)
        ttk.Label(input_frame, text="From (optional):").grid(row=2, column=0, sticky='w')
        self.from_entry = ttk.Entry(input_frame, width=80)
        self.from_entry.grid(row=2, column=1, padx=5, pady=5, sticky='ew')

        # Buttons
        button_frame = ttk.Frame(self.detection_tab)
        button_frame.pack(fill='x', padx=10, pady=5)

        self.detect_button = ttk.Button(button_frame, text="Detect Phishing",
                                        command=self.detect_phishing)
        self.detect_button.pack(side='left', padx=5)

        self.clear_button = ttk.Button(button_frame, text="Clear",
                                       command=self.clear_fields)
        self.clear_button.pack(side='left', padx=5)

        # Results frame
        self.results_frame = ttk.LabelFrame(self.detection_tab, text="Results", padding=10)
        self.results_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Configure grid weights
        input_frame.columnconfigure(1, weight=1)
        input_frame.rowconfigure(1, weight=1)

    def create_model_selection(self):
        """Add model selection to GUI"""
        # Add to detection tab (now detection_tab exists)
        self.model_frame = ttk.LabelFrame(self.detection_tab, text="Model Selection", padding=5)
        self.model_frame.pack(fill='x', padx=10, pady=5)

        self.model_var = tk.StringVar(value='Random Forest')

        ttk.Radiobutton(
            self.model_frame,
            text="Random Forest",
            variable=self.model_var,
            value='Random Forest',
            command=self.update_model
        ).pack(side='left', padx=10)

        ttk.Radiobutton(
            self.model_frame,
            text="XGBoost",
            variable=self.model_var,
            value='XGBoost',
            command=self.update_model
        ).pack(side='left', padx=10)

    def update_model(self):
        """Handle model selection change"""
        model_name = self.model_var.get()
        try:
            self.detector.set_model(model_name)
            messagebox.showinfo("Model Changed", f"Using {model_name} model")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_logo(self):
        """Load and display logo"""
        try:
            # Create a simple logo using text since we can't include image files
            logo_label = tk.Label(self.root, text="Phishing Detector",
                                  font=("Helvetica", 16, "bold"), fg="blue")
            logo_label.pack(pady=10)
        except Exception as e:
            logger.error(f"Could not load logo: {str(e)}")


    def create_training_tab(self):
        """Create the model training tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Train Model")

        # Training frame
        train_frame = ttk.LabelFrame(tab, text="Model Training", padding=10)
        train_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Instructions
        ttk.Label(train_frame,
                  text="Train a new model using a CSV file with email data.\n"
                       "The CSV should have at least 'body' and 'label' columns.\n"
                       "Label should be 0 for legitimate and 1 for phishing.").pack(pady=5)

        # File selection
        file_frame = ttk.Frame(train_frame)
        file_frame.pack(fill='x', pady=5)

        self.file_entry = ttk.Entry(file_frame)
        self.file_entry.pack(side='left', fill='x', expand=True, padx=5)

        browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_button.pack(side='left', padx=5)

        # Training buttons
        button_frame = ttk.Frame(train_frame)
        button_frame.pack(pady=10)

        self.train_button = ttk.Button(button_frame, text="Train Model",
                                       command=self.train_model)
        self.train_button.pack(side='left', padx=5)

        self.save_model_button = ttk.Button(button_frame, text="Save Model",
                                            command=self.save_model)
        self.save_model_button.pack(side='left', padx=5)

        self.load_model_button = ttk.Button(button_frame, text="Load Model",
                                            command=self.load_model)
        self.load_model_button.pack(side='left', padx=5)

        # Training results
        self.train_results = scrolledtext.ScrolledText(train_frame, height=10)
        self.train_results.pack(fill='both', expand=True, pady=5)
        self.train_results.insert('end', "Training results will appear here...")
        self.train_results.config(state='disabled')

    def create_about_tab(self):
        """Create the about tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="About")

        about_text = """
        Phishing Email Detector
        
        Hello, I'm Taiwo :)
        This application uses machine learning to detect phishing emails.
        It analyzes the content of emails to determine if they are legitimate or phishing attempts.

        Features to take note off for use:
        - Train on your own dataset (you can upload your csv dataset to train the model on).
        - Real-time phishing detection (After you train it, you can simply enter in the email information to inspect).
        - Explanation of predictions (Using explinable AI, I went with LIME & ELI5, you get details behind it's prediction).
        - Save and load trained models

        How to use:
        1. Train a model using a CSV file with labeled emails
        2. Use the detection tab to analyze emails
        3. View the results and explanations

        The model uses:
        - Random Forest classifier (No special reason on selection)
        - TF-IDF for text features
        - LIME and ELI5 for explanations (SHAP gave me a headache)
        """

        about_label = ttk.Label(tab, text=about_text, justify='left')
        about_label.pack(padx=10, pady=10, fill='both', expand=True)

    def browse_file(self):
        """Browse for CSV file"""
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            self.file_entry.delete(0, 'end')
            self.file_entry.insert(0, filename)

    def train_model(self):
        """Train the model"""
        model_name = self.model_var.get()
        filename = self.file_entry.get()

        if not filename:
            messagebox.showerror("Error", "Please select a CSV file first")
            return

        try:
            self.train_results.config(state='normal')
            self.train_results.delete(1.0, 'end')
            self.train_results.insert('end', f"Training {model_name} model... Please wait...\n")
            self.root.update()

            # Train the model
            y_test, y_pred, report = self.detector.train_from_csv(filename)

            # Display results
            self.train_results.insert('end', f"\n{model_name} training complete!\n\n")
            self.train_results.insert('end', "Classification Report:\n")
            self.train_results.insert('end', classification_report(y_test, y_pred))

            # Show accuracy
            accuracy = report['accuracy']
            self.train_results.insert('end', f"\nAccuracy: {accuracy * 100:.2f}%\n")

            messagebox.showinfo("Success", f"{model_name} model trained successfully!")

        except Exception as e:
            error_msg = f"Failed to train model: {str(e)}\n\n"
            error_msg += "Common issues:\n"
            error_msg += "1. Missing or invalid labels in dataset\n"
            error_msg += "2. Empty email bodies/subjects\n"
            error_msg += "3. Corrupted CSV file\n"
            error_msg += "4. Memory limits exceeded (try smaller dataset)\n"

            messagebox.showerror("Error", error_msg)
            self.train_results.insert('end', error_msg)
        finally:
            self.train_results.config(state='disabled')

    def save_model(self):
        """Save the trained model"""
        model_name = self.model_var.get()
        if not self.detector.is_trained:
            messagebox.showerror("Error", "No trained model to save")
            return

        try:
            directory = filedialog.askdirectory(title="Select directory to save model")
            if directory:
                # Save with model-specific filenames
                joblib.dump(
                    self.detector.model,
                    os.path.join(directory, f'{model_name.lower().replace(" ", "_")}_model.pkl')
                )
                joblib.dump(
                    self.detector.tfidf,
                    os.path.join(directory, 'tfidf_vectorizer.pkl')
                )

                # Save feature names
                with open(os.path.join(directory, 'feature_names.txt'), 'w', encoding='utf-8') as f:
                    for name in self.detector.feature_names:
                        f.write(f"{name}\n")

                messagebox.showinfo("Success", f"{model_name} model saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")

    def load_model(self):
        """Load a trained model"""
        try:
            directory = filedialog.askdirectory(title="Select directory containing model")
            if directory:
                # Try loading XGBoost first, then fall back to Random Forest
                model_path = os.path.join(directory, 'xgboost_model.pkl')
                if os.path.exists(model_path):
                    self.detector.set_model('XGBoost')
                    self.model_var.set('XGBoost')
                else:
                    model_path = os.path.join(directory, 'random_forest_model.pkl')
                    self.detector.set_model('Random Forest')
                    self.model_var.set('Random Forest')

                self.detector.model = joblib.load(model_path)
                self.detector.tfidf = joblib.load(os.path.join(directory, 'tfidf_vectorizer.pkl'))

                # Load feature names
                with open(os.path.join(directory, 'feature_names.txt'), 'r', encoding='utf-8') as f:
                    self.detector.feature_names = [line.strip() for line in f.readlines()]

                self.detector.is_trained = True
                messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")


    def detect_phishing(self):
        """Detect phishing in the entered email"""
        if not self.detector.is_trained:
            messagebox.showerror("Error", "Model not trained. Please train or load a model first.")
            return

        # Get email details
        subject = self.subject_entry.get()
        body = self.body_text.get("1.0", 'end-1c')
        from_addr = self.from_entry.get()

        if not subject and not body:
            messagebox.showerror("Error", "Please enter at least a subject or body")
            return

        try:
            # Clear previous results
            for widget in self.results_frame.winfo_children():
                widget.destroy()

            # Add loading message
            loading_label = ttk.Label(self.results_frame, text="Analyzing email...")
            loading_label.pack(pady=20)
            self.root.update()

            # Make prediction
            prediction, proba, explanation = self.detector.predict_email(subject, body)

            # Remove loading message
            loading_label.destroy()

            # Display results
            self.display_results(prediction, proba, explanation, subject, body, from_addr)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze email: {str(e)}")

    def display_results(self, prediction, proba, explanation, subject, body, from_addr):
        """Display the detection results"""
        # Result summary
        result_frame = ttk.Frame(self.results_frame)
        result_frame.pack(fill='x', pady=5)

        # Prediction
        prediction_text = "PHISHING" if prediction == 1 else "LEGITIMATE"
        color = "red" if prediction == 1 else "green"
        prediction_label = ttk.Label(result_frame,
                                     text=f"Prediction: {prediction_text}",
                                     font=("Helvetica", 14, "bold"),
                                     foreground=color)
        prediction_label.pack(side='left', padx=10)

        # Probability
        proba_text = f"Probability: {proba[prediction] * 100:.1f}%"
        proba_label = ttk.Label(result_frame,
                                text=proba_text,
                                font=("Helvetica", 12))
        proba_label.pack(side='left', padx=10)

        # Create notebook for detailed results
        results_notebook = ttk.Notebook(self.results_frame)
        results_notebook.pack(fill='both', expand=True, padx=10, pady=5)

        # Key features tab
        features_tab = ttk.Frame(results_notebook)
        results_notebook.add(features_tab, text="Key Features")
        self.display_key_features(features_tab, explanation['features'], prediction)

        # LIME explanation tab
        if explanation['lime']:
            lime_tab = ttk.Frame(results_notebook)
            results_notebook.add(lime_tab, text="LIME Explanation")
            self.display_html_explanation(lime_tab, explanation['lime'])

        # ELI5 explanation tab
        if explanation['eli5']:
            eli5_tab = ttk.Frame(results_notebook)
            results_notebook.add(eli5_tab, text="ELI5 Explanation")
            self.display_html_explanation(eli5_tab, explanation['eli5'])

        # Email preview tab
        preview_tab = ttk.Frame(results_notebook)
        results_notebook.add(preview_tab, text="Email Preview")
        self.display_email_preview(preview_tab, subject, body, from_addr)

    def display_key_features(self, parent, features, prediction):
        """Display the key features contributing to the prediction"""
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Title
        title = "Features indicating PHISHING" if prediction == 1 else "Features indicating LEGITIMATE"
        ttk.Label(frame, text=title, font=("Helvetica", 12, "bold")).pack(pady=5)

        # Create treeview
        tree = ttk.Treeview(frame, columns=('feature', 'value', 'contribution'), show='headings')
        tree.heading('feature', text='Feature')
        tree.heading('value', text='Value')
        tree.heading('contribution', text='Contribution')

        # Add data
        for feature in features:
            tree.insert('', 'end', values=(
                feature['feature'],
                f"{feature['value']:.4f}",
                f"{feature['contribution']:.4f}"
            ))

        tree.pack(fill='both', expand=True)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        scrollbar.pack(side='right', fill='y')
        tree.configure(yscrollcommand=scrollbar.set)

    def display_html_explanation(self, parent, html_content):
        """Display HTML explanation in a frame"""
        try:
            # Create temporary HTML file
            temp_file = "temp_explanation.html"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Open in browser button
            open_button = ttk.Button(parent,
                                     text="Open in Browser",
                                     command=lambda: webbrowser.open(temp_file))
            open_button.pack(pady=5)

            # Display HTML in a scrollable frame
            text = scrolledtext.ScrolledText(parent, wrap=tk.WORD)
            text.pack(fill='both', expand=True, padx=10, pady=5)

            # Convert HTML to plain text for display
            soup = BeautifulSoup(html_content, 'html.parser')
            text.insert('1.0', soup.get_text())
            text.config(state='disabled')

        except Exception as e:
            ttk.Label(parent, text=f"Could not display explanation: {str(e)}").pack()

    def display_email_preview(self, parent, subject, body, from_addr):
        """Display the original email content"""
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        # From
        if from_addr:
            ttk.Label(frame, text=f"From: {from_addr}", font=("Helvetica", 10)).pack(anchor='w')

        # Subject
        ttk.Label(frame, text=f"Subject: {subject}", font=("Helvetica", 10, "bold")).pack(anchor='w')

        # Body
        ttk.Label(frame, text="Body:", font=("Helvetica", 10)).pack(anchor='w')

        body_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
        body_text.pack(fill='both', expand=True)
        body_text.insert('1.0', body)
        body_text.config(state='disabled')

    def clear_fields(self):
        """Clear all input fields"""
        self.subject_entry.delete(0, 'end')
        self.body_text.delete('1.0', 'end')
        self.from_entry.delete(0, 'end')

        # Clear results
        for widget in self.results_frame.winfo_children():
            widget.destroy()


def main():
    root = tk.Tk()
    app = PhishingDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()