import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import mlflow
import mlflow.keras
import warnings

from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

# --------------------- Data Preparation ---------------------------- #
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df.drop(columns=[
        'other_combination_therapies', 
        'alpha_glucosidase_inhibitors',
        'meglitinides',
        'thiazolidinediones',
        'max_glu_serum',
        'binary_diabetesMed'
    ], inplace=True)
    
    X = df.drop(columns=['readmitted'])
    y = df['readmitted']
    
    return X, y

# --------------------- Resampling ---------------------------- #
def resample_data(X, y):
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    return X_resampled, y_resampled

# --------------------- Neural Network Modeling ---------------------------- #
def train_and_evaluate_nn(X_train, y_train, X_test, y_test, plot_name):
    mlflow.set_experiment("readmission-prediction")
    with mlflow.start_run(run_name="NeuralNetwork_Model") as run:
        model_name = f"NeuralNetwork_{plot_name}"

        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model building
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        # Evaluation
        loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
        mlflow.log_metrics({'test_loss': loss, 'test_accuracy': accuracy})
        mlflow.keras.log_model(model, f"models/{model_name}")

        print(f"Neural Network Test Accuracy: {accuracy:.2f}")

        # Accuracy plot
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        mlflow.log_figure(plt.gcf(), f'{model_name}_accuracy_plot.png')
        plt.close()

# --------------------- Main Function ---------------------------- #
def main(filepath):
    X, y = load_and_prepare_data(filepath)
    X_resampled, y_resampled = resample_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
    train_and_evaluate_nn(X_train, y_train, X_test, y_test, plot_name='smotetomek_nn')

# --------------------- Run Script ---------------------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', '-f', type=str, required=True, help="Path to the CSV file")
    args = parser.parse_args()
    main(filepath=args.filepath)
