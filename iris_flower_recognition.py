"""
==================================================================================
    IRIS FLOWER CLASSIFICATION SYSTEM
    Developed by: Dr. Mahroona Laraib
==================================================================================
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sys
import warnings
warnings.filterwarnings('ignore')

WELCOME_ART = r"""
    IRIS FLOWER CLASSIFICATION
    Dr. Mahroona Laraib
"""

MODEL_FILE = "mahroona_model.knn"
SCALER_FILE = "mahroona_scaler.bin"
ENCODER_FILE = "mahroona_encoder.bin"

def show_banner():
    """Display the application banner"""
    print("\033[95m" + WELCOME_ART + "\033[0m")
    print("\033[96m" + "="*50 + "\033[0m")

def check_existing_model():
    """Check if trained model already exists"""
    model_path = Path(MODEL_FILE)
    scaler_path = Path(SCALER_FILE)
    encoder_path = Path(ENCODER_FILE)
    
    if model_path.exists() and scaler_path.exists() and encoder_path.exists():
        return True
    return False

def load_model_files():
    """Load existing model and associated files"""
    print("\nFound existing model files...")
    print("Loading model from disk...")
    
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODER_FILE)
    
    print("Model loaded successfully!")
    return model, scaler, encoder

def prepare_dataset():
    """Load and prepare the iris dataset"""
    print("\nChecking for iris.csv...")
    
    if not Path('iris.csv').exists():
        print("\nERROR: iris.csv not found!")
        print("Please place the dataset in the current folder.")
        print("Expected format: Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species")
        sys.exit(1)
    
    print("Reading dataset...")
    data = pd.read_csv('iris.csv')
    print(f"Loaded {len(data)} flower samples")
    
    return data

def clean_data(data_frame):
    """Remove unnecessary columns and separate features/target"""
    print("\nCleaning data...")
    
    if 'Id' in data_frame.columns:
        features = data_frame.drop(['Id', 'Species'], axis=1)
        print("Removed ID column")
    else:
        features = data_frame.drop(['Species'], axis=1)
    
    targets = data_frame['Species']
    
    print(f"Features: {list(features.columns)}")
    print(f"Target: {targets.name}")
    
    return features, targets

def encode_species(targets):
    """Convert species names to numbers"""
    print("\nEncoding species names...")
    
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(targets)
    
    print("Species codebook:")
    for i, name in enumerate(encoder.classes_):
        print(f"   {i} → {name}")
    
    return encoded, encoder

def normalize_features(features):
    """Scale features to have zero mean and unit variance"""
    print("\nNormalizing measurements...")
    
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features)
    
    print("Feature scaling complete")
    print(f"   Mean after scaling: {normalized.mean(axis=0).round(2)}")
    print(f"   Std after scaling: {normalized.std(axis=0).round(2)}")
    
    return normalized, scaler

def split_dataset(normalized_data, encoded_targets):
    """Split data into training and testing sets"""
    print("\nCreating train/test split...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_data, encoded_targets, 
        test_size=0.3, 
        random_state=42, 
        stratify=encoded_targets
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def find_best_k(X_train, X_test, y_train, y_test):
    """Find optimal k value for KNN"""
    print("\n🐢 Searching for best k value...")
    print("   Testing k from 1 to 15...")
    
    k_range = range(1, 16)
    k_scores = []
    
    for k in k_range:
        temp_model = KNeighborsClassifier(n_neighbors=k)
        temp_model.fit(X_train, y_train)
        predictions = temp_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        k_scores.append(accuracy)
        print(f"   k={k:2d}: {accuracy*100:.2f}%")
    
    best_index = np.argmax(k_scores)
    best_k = k_range[best_index]
    best_score = k_scores[best_index]
    
    print(f"\n🐢 Best k = {best_k} with {best_score*100:.2f}% accuracy")
    
    return best_k, best_score

def train_final_model(X_train, y_train, best_k):
    """Train final KNN model with optimal k"""
    print("\nTraining final model...")
    
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)
    
    print("Training complete!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set"""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nTest set accuracy: {accuracy*100:.2f}%")
    
    return accuracy

def save_model_artifacts(model, scaler, encoder):
    """Save trained model and associated files"""
    print("\nSaving model for future use...")
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(encoder, ENCODER_FILE)
    
    print(f"Model saved as: {MODEL_FILE}")
    print(f"Scaler saved as: {SCALER_FILE}")
    print(f"Encoder saved as: {ENCODER_FILE}")

def predict_species(sepal_len, sepal_wid, petal_len, petal_wid, model, scaler, encoder):
    """Predict species for a single flower"""
    flower = pd.DataFrame({
        'SepalLengthCm': [sepal_len],
        'SepalWidthCm': [sepal_wid],
        'PetalLengthCm': [petal_len],
        'PetalWidthCm': [petal_wid]
    })
    
    flower_scaled = scaler.transform(flower)
    prediction_idx = model.predict(flower_scaled)[0]
    probabilities = model.predict_proba(flower_scaled)[0]
    
    species = encoder.inverse_transform([prediction_idx])[0]
    confidence = probabilities[prediction_idx] * 100
    
    return species, confidence, probabilities

def run_interactive_mode(model, scaler, encoder):
    """Run interactive prediction session"""
    print("INTERACTIVE PREDICTION MODE")
    print("Type 'exit' to quit")

    
    session_count = 0
    
    while True:
        session_count += 1
        print(f"\nFlower #{session_count}")
        print("─" * 30)
        
        try:
            sl = input("Sepal Length (cm): ").strip()
            if sl.lower() == 'exit': break
            sl = float(sl)
            
            sw = input("Sepal Width (cm): ").strip()
            if sw.lower() == 'exit': break
            sw = float(sw)
            
            pl = input("Petal Length (cm): ").strip()
            if pl.lower() == 'exit': break
            pl = float(pl)
            
            pw = input("Petal Width (cm): ").strip()
            if pw.lower() == 'exit': break
            pw = float(pw)
            
        except ValueError:
            print("❌ Please enter valid numbers!")
            continue
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        species, confidence, probs = predict_species(sl, sw, pl, pw, model, scaler, encoder)
        
        print(f"   {species}")
        print(f"   Confidence: {confidence:.1f}%")
        print("─" * 30)
        
        print("   Probability Breakdown:")
        for i, sp in enumerate(encoder.classes_):
            bar = "▓" * int(probs[i] * 20) + "░" * (20 - int(probs[i] * 20))
            print(f"   {sp[:4]}: {bar} {probs[i]*100:5.1f}%")
        
        print("─" * 30)

def main():
    """Main program execution"""
    show_banner()
    
    if check_existing_model():
        knn_model, feature_scaler, species_encoder = load_model_files()
        print("\nReady to make predictions!")
    else:
        print("\nNo existing model found. Starting training...")
        
        raw_data = prepare_dataset()
        clean_features, clean_targets = clean_data(raw_data)
        encoded_targets, species_encoder = encode_species(clean_targets)
        normalized_features, feature_scaler = normalize_features(clean_features)
        X_train, X_test, y_train, y_test = split_dataset(normalized_features, encoded_targets)
        best_k, _ = find_best_k(X_train, X_test, y_train, y_test)
        knn_model = train_final_model(X_train, y_train, best_k)
        evaluate_model(knn_model, X_test, y_test)
        save_model_artifacts(knn_model, feature_scaler, species_encoder)

    user_choice = input("Start interactive mode? (y/n): ").strip().lower()
    
    if user_choice == 'y':
        run_interactive_mode(knn_model, feature_scaler, species_encoder)
    
    print("\n" + "="*50)
    print("Program completed - Dr. Mahroona Laraib")
    print("="*50)

if __name__ == "__main__":
    main()