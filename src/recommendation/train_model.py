from src.recommendation.utils.utils import *
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, r2_score, recall_score, accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
import joblib
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.preprocessing import MinMaxRewardScaler


def train_model(method, is_regressor, method_model=None, threshold=None, data='T0'):
    DATA_PATH, MODEL_DIR, METRICS_DIR, OPTIMAL_THRS = train_model_path_indicator(method, is_regressor, method_model, threshold, data)
    print(MODEL_DIR)
    print(METRICS_DIR)
    os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_DIR), exist_ok=True)

    if method == "reinforcement_learning":
        """Train offline RL model with transaction count rewards"""
        
        # Load dataset
        try:
            dataset = MDPDataset.load(f'{DATA_PATH}/mdp_dataset.h5')
        except:
            dataset = load_dataset_components(DATA_PATH)
        
        reward_scaler = MinMaxRewardScaler(minimum=0.0, maximum=1.0)

        # Initialize CQL with continuous reward support
        cql = DiscreteCQLConfig(
            learning_rate=6.25e-05,
            batch_size=32,
            # n_critics=2,
            # alpha=1.5,
            reward_scaler=reward_scaler  # Important for continuous rewards
        ).create(device=None)
        
        cql.build_with_dataset(dataset)
        
        # Training
        print("Starting CQL training with transaction count rewards...")
        cql.fit(
            dataset,
            n_steps=200000,
            save_interval=10000,
        )
        
        # Save model
        model_path = f'{MODEL_DIR}/cql_model_txn_counts{OPTIMAL_THRS}.d3'
        cql.save_model(model_path)
        print(f"✅ Model saved to {model_path}")
        
        return cql


    else:
        df, preprocessor = load_and_preprocess_data(DATA_PATH)

        print(DATA_PATH)
        
        # Get feature columns (exclude target categories)
        # feature_cols = [col for col in df.columns if col not in categories and col != 'CUST_ID']
        X_df = df[feature_cols]
        
        # Fit preprocessor on feature columns only
        X = preprocessor.fit_transform(X_df)
        results = {}

        if method == "binary":
            optimal_thresholds = {}

            for category in categories:
                if category not in df.columns:
                    print(f"Warning: Category '{category}' not found in data. Skipping...")
                    continue
                    
                print(f"\nTraining model for {category}...")
                y = df[category]

                # Skip if all zeros (no transactions)
                if y.sum() == 0:
                    print(f"Warning: Category '{category}' has no transactions. Skipping...")
                    continue


                if is_regressor:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
                else:
                    # # Check class distribution for stratification
                    # class_counts = y.value_counts()
                    # min_class_count = class_counts.min()

                    # # Check if we have both classes
                    # unique_classes = y.unique()
                    # if len(unique_classes) < 2:
                    #     print(f"Warning: Category '{category}' has only one class. Skipping...")
                    #     continue
                    
                    # # If any class has fewer than 2 samples, we can't use stratification
                    # if min_class_count < 2:
                    #     print(f"Warning: Category '{category}' has classes with too few samples (min: {min_class_count}). Using random split...")
                    #     X_train, X_test, y_train, y_test = train_test_split(
                    #         X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
                    # else:
                    #     # Safe to use stratification
                    #     X_train, X_test, y_train, y_test = train_test_split(
                    #         X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

                    y_binary = (y > 0).astype(int)
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_binary, test_size=TEST_SIZE, random_state=RANDOM_STATE)

                
                # binary classifications
                if method_model == "random_forests":
                    if is_regressor:
                        reg = RandomForestRegressor(
                            random_state=RANDOM_STATE,
                            n_jobs=-1
                        )
                        reg.fit(X_train, y_train)
                        
                        # Evaluate
                        y_pred = reg.predict(X_test)

                        if threshold is not None:
                            # Regression metrics
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # Print metrics
                            print(f"  RMSE: {rmse:.4f}")
                            print(f"  MAE: {mae:.4f}")
                            print(f"  R²: {r2:.4f}")
                            
                            results[category] = {
                                'rmse': rmse,
                                'mae': mae,
                                'r2': r2
                            }
                            
                            # Save model with metadata
                            model_data = {
                                'model': reg,
                                'category': category,
                                'preprocessor': preprocessor,
                                'is_regression': True
                            }
                        else:
                            # Find optimal threshold for this category
                            optimal_threshold = find_optimal_regression_threshold(y_test, y_pred)
                            optimal_thresholds[category] = optimal_threshold
                            
                            # Evaluate with optimal threshold
                            y_pred_binary = (y_pred > optimal_threshold).astype(int)
                            y_true_binary = (y_test > 0).astype(int)
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_true_binary, y_pred_binary)
                            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                            
                            # Regression metrics
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            print(f"  Optimal threshold: {optimal_threshold:.4f}")
                            print(f"  Accuracy: {accuracy:.4f}")
                            print(f"  Precision: {precision:.4f}")
                            print(f"  Recall: {recall:.4f}")
                            print(f"  F1: {f1:.4f}")
                            print(f"  RMSE: {rmse:.4f}")
                            print(f"  MAE: {mae:.4f}")
                            print(f"  R²: {r2:.4f}")
                            
                            results[category] = {
                                'optimal_threshold': optimal_threshold,
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'rmse': rmse,
                                'mae': mae,
                                'r2': r2
                            }
                            
                            # Save model with metadata including optimal threshold
                            model_data = {
                                'model': reg,
                                'category': category,
                                'preprocessor': preprocessor,
                                'is_regression': True,
                                'optimal_threshold': optimal_threshold
                            }
                        
                        joblib.dump(model_data, f"{MODEL_DIR}/{category}_model{OPTIMAL_THRS}.pkl")

                    else:
                        clf = RandomForestClassifier(
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            class_weight='balanced'
                        )
                        clf.fit(X_train, y_train)

                        if threshold is not None:
                            # Evaluate with better handling of warnings
                            y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Get probabilities for class 1
                            y_pred = (y_pred_proba > threshold).astype(int)

                            # Classification metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, zero_division=0)
                            recall = recall_score(y_test, y_pred, zero_division=0)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            roc_auc = roc_auc_score(y_test, y_pred_proba)
                            
                            # Print metrics
                            print(f"  Accuracy: {accuracy:.4f}")
                            print(f"  Precision: {precision:.4f}")
                            print(f"  Recall: {recall:.4f}")
                            print(f"  F1: {f1:.4f}")
                            print(f"  ROC AUC: {roc_auc:.4f}")
                            
                            results[category] = {
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'roc_auc': roc_auc
                            }
                            
                            # Save model with metadata
                            model_data = {
                                'model': clf,
                                'category': category,
                                'preprocessor': preprocessor,
                                'is_regression': False
                            }

                        else:
                            # Get probabilities on validation set
                            y_val_proba = clf.predict_proba(X_test)[:, 1]
                            # Find optimal threshold for this category
                            optimal_threshold = find_optimal_classification_threshold(y_test, y_val_proba)
                            optimal_thresholds[category] = optimal_threshold
                            
                            # Evaluate with optimal threshold
                            y_pred = (y_val_proba > optimal_threshold).astype(int)
                            
                            # Classification metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, zero_division=0)
                            recall = recall_score(y_test, y_pred, zero_division=0)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            roc_auc = roc_auc_score(y_test, y_val_proba)
                            
                            print(f"  Optimal threshold: {optimal_threshold:.4f}")
                            print(f"  Accuracy: {accuracy:.4f}")
                            print(f"  Precision: {precision:.4f}")
                            print(f"  Recall: {recall:.4f}")
                            print(f"  F1: {f1:.4f}")
                            print(f"  ROC AUC: {roc_auc:.4f}")
                            
                            results[category] = {
                                'optimal_threshold': optimal_threshold,
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'roc_auc': roc_auc
                            }
                            
                            # Save model with metadata including optimal threshold
                            model_data = {
                                'model': clf,
                                'category': category,
                                'preprocessor': preprocessor,
                                'is_regression': False,
                                'optimal_threshold': optimal_threshold
                            }
                        
                        
                        joblib.dump(model_data, f"{MODEL_DIR}/{category}_model{OPTIMAL_THRS}.pkl")


                elif method_model == "catboost":
                    if is_regressor:
                        model = CatBoostRegressor(
                            iterations=1000,
                            learning_rate=0.1,
                            depth=6,
                            random_seed=RANDOM_STATE,
                            verbose=100,  # Shows progress every 100 iterations
                            task_type='CPU'
                        )
                            
                        model.fit(
                            X_train, y_train,
                            eval_set=(X_test, y_test),
                            early_stopping_rounds=50
                        )
                        
                        # Evaluate
                        y_pred = model.predict(X_test)

                        if threshold != None:
                            # Regression metrics
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            # Print metrics
                            print(f"  RMSE: {rmse:.4f}")
                            print(f"  MAE: {mae:.4f}")
                            print(f"  R²: {r2:.4f}")
                            
                            results[category] = {
                                'rmse': rmse,
                                'mae': mae,
                                'r2': r2
                            }
                            
                            # Save model with metadata
                            model_data = {
                                'model': model,
                                'category': category,
                                'preprocessor': preprocessor,
                                'is_regression': True
                            }
                            
                        else:
                            # Find optimal threshold for this category
                            optimal_threshold = find_optimal_regression_threshold(y_test, y_pred)
                            optimal_thresholds[category] = optimal_threshold
                            
                            # Evaluate with optimal threshold
                            y_pred_binary = (y_pred > optimal_threshold).astype(int)
                            y_true_binary = (y_test > 0).astype(int)
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_true_binary, y_pred_binary)
                            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                            
                            # Regression metrics
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            print(f"  Optimal threshold: {optimal_threshold:.4f}")
                            print(f"  Accuracy: {accuracy:.4f}")
                            print(f"  Precision: {precision:.4f}")
                            print(f"  Recall: {recall:.4f}")
                            print(f"  F1: {f1:.4f}")
                            print(f"  RMSE: {rmse:.4f}")
                            print(f"  MAE: {mae:.4f}")
                            print(f"  R²: {r2:.4f}")
                            
                            results[category] = {
                                'optimal_threshold': optimal_threshold,
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'rmse': rmse,
                                'mae': mae,
                                'r2': r2
                            }
                            
                            # Save model with metadata including optimal threshold
                            model_data = {
                                'model': model,
                                'category': category,
                                'preprocessor': preprocessor,
                                'is_regression': True,
                                'optimal_threshold': optimal_threshold
                            }
                            
                        joblib.dump(model_data, f"{MODEL_DIR}/{category}_model{OPTIMAL_THRS}.pkl")
                    else:
                        model = CatBoostClassifier(
                            iterations=1000,
                            learning_rate=0.1,
                            depth=6,
                            random_seed=RANDOM_STATE,
                            verbose=100,
                            task_type='CPU',
                            loss_function='Logloss',
                            eval_metric='AUC'
                        )
                        
                        model.fit(
                            X_train, y_train,
                            eval_set=(X_test, y_test),
                            early_stopping_rounds=50
                        )
                        

                        if threshold != None:
                            # Evaluate
                            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1
                            y_pred = (y_pred_proba > threshold).astype(int)  # Convert to binary predictions
                            
                            # Classification metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, zero_division=0)
                            recall = recall_score(y_test, y_pred, zero_division=0)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            roc_auc = roc_auc_score(y_test, y_pred_proba)
                            
                            # Print metrics
                            print(f"  Accuracy: {accuracy:.4f}")
                            print(f"  Precision: {precision:.4f}")
                            print(f"  Recall: {recall:.4f}")
                            print(f"  F1: {f1:.4f}")
                            print(f"  ROC AUC: {roc_auc:.4f}")
                            
                            results[category] = {
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'roc_auc': roc_auc
                            }
                            
                            # Save model with metadata
                            model_data = {
                                'model': model,
                                'category': category,
                                'preprocessor': preprocessor,
                                'is_regression': False
                            }
                            
                        else:
                            # Get probabilities on validation set
                            y_val_proba = model.predict_proba(X_test)[:, 1]
                            # Find optimal threshold for this category
                            optimal_threshold = find_optimal_classification_threshold(y_test, y_val_proba)
                            optimal_thresholds[category] = optimal_threshold
                            
                            # Evaluate with optimal threshold
                            y_pred = (y_val_proba > optimal_threshold).astype(int)
                            
                            # Classification metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, zero_division=0)
                            recall = recall_score(y_test, y_pred, zero_division=0)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            roc_auc = roc_auc_score(y_test, y_val_proba)
                            
                            print(f"  Optimal threshold: {optimal_threshold:.4f}")
                            print(f"  Accuracy: {accuracy:.4f}")
                            print(f"  Precision: {precision:.4f}")
                            print(f"  Recall: {recall:.4f}")
                            print(f"  F1: {f1:.4f}")
                            print(f"  ROC AUC: {roc_auc:.4f}")
                            
                            results[category] = {
                                'optimal_threshold': optimal_threshold,
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'roc_auc': roc_auc
                            }
                            
                            # Save model with metadata including optimal threshold
                            model_data = {
                                'model': model,
                                'category': category,
                                'preprocessor': preprocessor,
                                'is_regression': False,
                                'optimal_threshold': optimal_threshold
                            }
                        
                        joblib.dump(model_data, f"{MODEL_DIR}/{category}_model{OPTIMAL_THRS}.pkl")


            if threshold is None:
                with open(f"{MODEL_DIR}/optimal_thresholds.json", 'w') as f:
                    json.dump(optimal_thresholds, f, indent=2)
                
            joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor{OPTIMAL_THRS}.pkl")
            


        # multilabel
        elif method == "multilabel":
            y = df[categories].values
        
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            
            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            if method_model == "multioutputclassifier":
                # Initialize base classifier
                catboost = CatBoostClassifier(
                    iterations=1000,
                    learning_rate=0.1,
                    depth=6,
                    random_seed=RANDOM_STATE,
                    verbose=100,
                    task_type='CPU'
                )

                # Create MultiOutputClassifier
                classifier = MultiOutputClassifier(catboost)
                
                # Create pipeline
                pipeline = Pipeline([
                    ('classifier', classifier)
                ])
                
                # Train model
                print("Training MultiOutput CatBoost model...")
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                # y_pred_binary = pipeline.predict(X_test)
                # y_pred_proba = pipeline.predict_proba(X_test)
                y_pred_proba = np.array([est.predict_proba(X_test)[:, 1] for est in pipeline.steps[-1][1].estimators_]).T

                # Handle thresholding
                if threshold is not None:
                    y_pred_binary = (y_pred_proba > threshold).astype(int)
                    optimal_thresholds = {cat: threshold for cat in categories}
                else:
                    optimal_thresholds = {}
                    y_pred_binary = np.zeros_like(y_pred_proba)
                    for i, cat in enumerate(categories):
                        optimal_threshold = find_optimal_classification_threshold(y_test[:, i], y_pred_proba[:, i])
                        optimal_thresholds[cat] = float(optimal_threshold)
                        y_pred_binary[:, i] = (y_pred_proba[:, i] > optimal_threshold).astype(int)
                
                # Save model and preprocessing components
                joblib.dump({
                    'model': pipeline,
                    'preprocessor': preprocessor,
                    'scaler': scaler,
                    'categories': categories,
                    'optimal_thresholds': optimal_thresholds,
                    'is_regression': False
                }, f"{MODEL_DIR}/multioutput_model{OPTIMAL_THRS}.pkl")

            elif method_model == "neural_network":
                # Create datasets and dataloaders
                train_dataset = TransactionDataset(X_train, y_train)
                test_dataset = TransactionDataset(X_test, y_test)
                
                batch_size = 64
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size)
                
                # Initialize model
                input_size = X_train.shape[1]
                output_size = len(categories)
                model = MultiLabelClassifier(input_size, output_size)
                
                # Define loss and optimizer
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Training loop
                num_epochs = 50
                best_loss = float('inf')
                
                for epoch in range(num_epochs):
                    model.train()
                    running_loss = 0.0
                    
                    for batch in train_loader:
                        features = batch['features']
                        labels = batch['labels']
                        
                        # Zero the parameter gradients
                        optimizer.zero_grad()
                        
                        # Forward pass
                        outputs = model(features)
                        loss = criterion(outputs, labels)
                        
                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()
                        
                        running_loss += loss.item()
                    
                    # Validation
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch in test_loader:
                            features = batch['features']
                            labels = batch['labels']
                            outputs = model(features)
                            val_loss += criterion(outputs, labels).item()
                    
                    avg_train_loss = running_loss / len(train_loader)
                    avg_val_loss = val_loss / len(test_loader)
                    
                    print(f'Epoch {epoch+1}/{num_epochs}, '
                        f'Train Loss: {avg_train_loss:.4f}, '
                        f'Val Loss: {avg_val_loss:.4f}')
                    
                    # Save best model
                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        # Save model weights only (safest option)
                        torch.save(model.state_dict(), f"{MODEL_DIR}/best_model_weights{OPTIMAL_THRS}.pth")
                        
                        # Save metadata separately
                        joblib.dump({
                            'epoch': epoch,
                            'categories': categories,
                            'input_size': input_size,
                            'output_size': output_size,
                            'scaler_mean': scaler.mean_,
                            'scaler_scale': scaler.scale_,
                        }, f"{MODEL_DIR}/model_metadata{OPTIMAL_THRS}.pkl")
                        
                        # Save preprocessor and scaler
                        joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor{OPTIMAL_THRS}.pkl")
                        joblib.dump(scaler, f"{MODEL_DIR}/scaler{OPTIMAL_THRS}.pkl")
                
                # Load best model
                model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model_weights{OPTIMAL_THRS}.pth"))
                model.eval()

                # Calculate metrics
                y_pred = []
                y_true = []
                with torch.no_grad():
                    for batch in test_loader:
                        features = batch['features']
                        labels = batch['labels']
                        outputs = model(features)
                        y_pred.extend(outputs.numpy())
                        y_true.extend(labels.numpy())
                
                y_pred = np.array(y_pred)
                y_true = np.array(y_true)
                
                # Handle thresholding
                if threshold is not None:
                    y_pred_binary = (y_pred > threshold).astype(int)
                    optimal_thresholds = {cat: threshold for cat in categories}
                else:
                    # Find optimal thresholds per category
                    optimal_thresholds = {}
                    y_pred_binary = np.zeros_like(y_pred)
                    for i, cat in enumerate(categories):
                        optimal_threshold = find_optimal_classification_threshold(y_true[:, i], y_pred[:, i])
                        optimal_thresholds[cat] = float(optimal_threshold)
                        y_pred_binary[:, i] = (y_pred[:, i] > optimal_threshold).astype(int)
                
                # Save optimal thresholds
                with open(f"{MODEL_DIR}/optimal_thresholds.json", 'w') as f:
                    json.dump(optimal_thresholds, f, indent=2)
                

            # Calculate metrics for each category
            for i, category in enumerate(categories):
                # # Get probabilities for positive class (assuming binary classification)
                # proba = y_pred_proba[i][:, 1] if len(y_pred_proba[i].shape) > 1 else y_pred_proba[i]
                
                tp = np.sum((y_pred_binary[:, i] == 1) & (y_test[:, i] == 1))
                fp = np.sum((y_pred_binary[:, i] == 1) & (y_test[:, i] == 0))
                fn = np.sum((y_pred_binary[:, i] == 0) & (y_test[:, i] == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                results[category] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'optimal_threshold': optimal_thresholds.get(category, threshold)
                }

        # Ensure METRICS_DIR exists
        os.makedirs(METRICS_DIR, exist_ok=True)

        with open(f"{METRICS_DIR}/training_metrics{OPTIMAL_THRS}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nTraining completed. Models and metrics saved.")
        
        if results:
            print("\nTraining Metrics Summary:")
            results_df = pd.DataFrame(results).T
            print(results_df.round(4))
        else:
            print("No models were trained successfully.")





if __name__ == "__main__":
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    categories = ['loan','utility','finance','shopping','financial_services', 'health_and_care', 'home_lifestyle', 'transport_travel',	
                 'leisure', 'public_services']
    feature_cols = ['Number of Children', 'Age', 'Gender', 'Education level', 'Marital status', 'Region', 'Occupation Group']
    
    # train_model(method="binary", is_regressor=True, method_model="random_forests", threshold=None)
    # train_model(method="binary", is_regressor=False, method_model="random_forests", threshold=None)
    # train_model(method="binary", is_regressor=True, method_model="random_forests", threshold=0)
    # train_model(method="binary", is_regressor=True, method_model="catboost", threshold=None)
    # train_model(method="binary", is_regressor=False, method_model="catboost", threshold=None)
    # train_model(method="binary", is_regressor=True, method_model="catboost", threshold=0)

    # train_model(method="multilabel", is_regressor=False, method_model="multioutputclassifier", threshold=None)
    # train_model(method="multilabel", is_regressor=False, method_model="neural_network", threshold=None)

    # train_model(method="reinforcement_learning", is_regressor=False, method_model=None, threshold=None)


    # T0/T1/predT1
    # train_model(method="binary", is_regressor=False, method_model="catboost", threshold=None)
    # train_model(method="binary", is_regressor=False, method_model="catboost", threshold=None, data='T1')
    train_model(method="binary", is_regressor=False, method_model="catboost", threshold=None, data='T1_predicted')

    # train_model(method="multilabel", is_regressor=False, method_model="neural_network", threshold=None, data='T1')
    # train_model(method="multilabel", is_regressor=False, method_model="neural_network", threshold=None, data='T1_predicted')