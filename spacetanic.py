import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
# import lightgbm as lgb  # Commented out for now
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint
from sklearn.decomposition import PCA

# Load the training data
df = pd.read_csv('train.csv')

# Save PassengerId for later use
passenger_ids = df['PassengerId'].copy()

# Enhanced feature engineering
def enhanced_feature_engineering(df):
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Extract group and number from PassengerId
    df_processed['Group'] = df_processed['PassengerId'].str.split('_').str[0]
    df_processed['GroupSize'] = df_processed.groupby('Group')['Group'].transform('count')
    df_processed['IsAlone'] = (df_processed['GroupSize'] == 1).astype(int)
    
    # Split cabin into deck, number, and side
    df_processed[['Deck', 'CabinNum', 'Side']] = df_processed['Cabin'].str.split('/', expand=True)
    df_processed['CabinNum'] = pd.to_numeric(df_processed['CabinNum'], errors='coerce')
    
    # Create location tracking features
    # First, create boolean columns for each spending location
    spending_locations = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for loc in spending_locations:
        df_processed[f'WasAt_{loc}'] = (df_processed[loc].fillna(0) > 0).astype(int)
    
    # Create cabin location features
    decks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
    for deck in decks:
        df_processed[f'Cabin_Deck_{deck}'] = (df_processed['Deck'] == deck).astype(int)
    
    # Create boolean columns for port/starboard
    df_processed['Cabin_Port'] = (df_processed['Side'] == 'P').astype(int)
    df_processed['Cabin_Starboard'] = (df_processed['Side'] == 'S').astype(int)
    
    # Create cabin number ranges
    cabin_ranges = [(0, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
    for start, end in cabin_ranges:
        df_processed[f'Cabin_Num_{start}_{end}'] = ((df_processed['CabinNum'] >= start) & 
                                                   (df_processed['CabinNum'] <= end)).astype(int)
    
    # Handle cryosleep passengers
    cryosleep_mask = df_processed['CryoSleep'].fillna(False)
    
    # Reset all spending location features to 0 for cryosleep passengers
    spending_location_cols = [col for col in df_processed.columns if col.startswith('WasAt_')]
    for col in spending_location_cols:
        df_processed.loc[cryosleep_mask, col] = 0
    
    # Create deck features
    df_processed['DeckLevel'] = df_processed['Deck'].map({
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8
    })
    
    # Create spending features
    df_processed['TotalSpending'] = df_processed[spending_locations].fillna(0).sum(axis=1)
    df_processed['SpendingPerPerson'] = df_processed['TotalSpending'] / df_processed['GroupSize'].replace(0, 1)
    df_processed['HasSpent'] = (df_processed['TotalSpending'] > 0).astype(int)
    
    # Create spending ratios
    for col in spending_locations:
        df_processed[f'{col}Ratio'] = df_processed[col].fillna(0) / df_processed['TotalSpending'].replace(0, 1)
    
    # Create age features
    df_processed['AgeGroup'] = pd.cut(df_processed['Age'].fillna(df_processed['Age'].median()), 
                                    bins=[0, 12, 18, 30, 50, 100],
                                    labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    df_processed['IsChild'] = (df_processed['Age'].fillna(df_processed['Age'].median()) <= 12).astype(int)
    df_processed['IsElderly'] = (df_processed['Age'].fillna(df_processed['Age'].median()) >= 50).astype(int)
    
    # Create family features
    df_processed['FamilySize'] = df_processed['GroupSize']
    df_processed['IsFamily'] = (df_processed['GroupSize'] > 1).astype(int)
    
    # Create luxury features
    df_processed['IsVIP'] = df_processed['VIP'].fillna(False).astype(int)
    df_processed['IsCryoSleep'] = df_processed['CryoSleep'].fillna(False).astype(int)
    df_processed['LuxuryScore'] = (
        df_processed['IsVIP'] * 2 + 
        df_processed['TotalSpending'] / df_processed['TotalSpending'].replace(0, 1) * 3
    )
    
    # Create destination features
    df_processed['IsHomePlanet'] = (df_processed['HomePlanet'] == df_processed['Destination']).astype(int)
    
    # Create cabin features
    df_processed['CabinSide'] = df_processed['Side'].map({'P': 0, 'S': 1}).fillna(0)
    df_processed['CabinDeck'] = df_processed['Deck'].map({
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8
    }).fillna(0)
    
    # Create meaningful interaction features
    # Cryosleep with location
    df_processed['CryoSleep_Deck'] = df_processed['IsCryoSleep'] * df_processed['CabinDeck']
    df_processed['CryoSleep_Side'] = df_processed['IsCryoSleep'] * df_processed['CabinSide']
    
    # VIP with location
    df_processed['VIP_Deck'] = df_processed['IsVIP'] * df_processed['CabinDeck']
    df_processed['VIP_Side'] = df_processed['IsVIP'] * df_processed['CabinSide']
    
    # Family with location
    df_processed['Family_Deck'] = df_processed['IsFamily'] * df_processed['CabinDeck']
    df_processed['Family_Side'] = df_processed['IsFamily'] * df_processed['CabinSide']
    
    # Create composite features
    df_processed['Location_Type'] = df_processed['CabinDeck'].astype(str) + '_' + df_processed['CabinSide'].astype(str)
    df_processed['CryoSleep_Location_Type'] = df_processed['IsCryoSleep'].astype(str) + '_' + df_processed['Location_Type']
    
    # Create spending patterns
    df_processed['Spending_Pattern'] = df_processed[spending_locations].fillna(0).apply(
        lambda x: '_'.join(x.astype(int).astype(str)), axis=1
    )
    
    # Drop only the original columns we don't need anymore
    columns_to_drop = ['PassengerId', 'Name', 'Cabin', 'Group']
    df_processed = df_processed.drop(columns=columns_to_drop)
    
    return df_processed

# Handle missing values with improved strategies
def handle_missing_values(df):
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Handle numerical columns with specific strategies
    numerical_cols = {
        'Age': 'median',  # Age is better handled with median
        'RoomService': 'zero',  # No spending means 0
        'FoodCourt': 'zero',
        'ShoppingMall': 'zero',
        'Spa': 'zero',
        'VRDeck': 'zero',
        'CabinNum': 'median',  # Cabin number should use median
        'TotalSpending': 'zero',
        'SpendingPerPerson': 'zero',
        'GroupSize': 'mode',  # Group size should use mode
        'DeckLevel': 'mode',
        'LuxuryScore': 'zero',
        'CabinSide': 'mode',
        'CabinDeck': 'mode',
        'CryoSleep_Deck': 'zero',
        'CryoSleep_Side': 'zero',
        'VIP_Deck': 'zero',
        'VIP_Side': 'zero',
        'Family_Deck': 'zero',
        'Family_Side': 'zero'
    }
    
    # Apply specific imputation strategies
    for col, strategy in numerical_cols.items():
        if col in df_processed.columns:
            if strategy == 'median':
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            elif strategy == 'mode':
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            elif strategy == 'zero':
                df_processed[col] = df_processed[col].fillna(0)
    
    # Handle spending ratio columns
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in spending_cols:
        ratio_col = f'{col}Ratio'
        if ratio_col in df_processed.columns:
            # For spending ratios, if total spending is 0, set ratio to 0
            df_processed[ratio_col] = df_processed[ratio_col].fillna(0)
    
    # Handle categorical columns with specific strategies
    categorical_cols = {
        'HomePlanet': 'mode',  # Most common home planet
        'CryoSleep': 'mode',  # Most common cryosleep status
        'Deck': 'mode',  # Most common deck
        'Side': 'mode',  # Most common side
        'Destination': 'mode',  # Most common destination
        'VIP': 'mode',  # Most common VIP status
        'AgeGroup': 'mode',  # Most common age group
        'Location_Type': 'mode',  # Most common location type
        'CryoSleep_Location_Type': 'mode',  # Most common cryosleep location
        'Spending_Pattern': 'mode'  # Most common spending pattern
    }
    
    # Apply specific imputation strategies for categorical columns
    for col, strategy in categorical_cols.items():
        if col in df_processed.columns:
            if strategy == 'mode':
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    # Create missing value indicators for important features
    important_cols = ['CryoSleep', 'VIP', 'HomePlanet', 'Destination']
    for col in important_cols:
        if col in df_processed.columns:
            df_processed[f'{col}_Missing'] = df_processed[col].isna().astype(int)
    
    # Create interaction features for missing values
    if 'CryoSleep_Missing' in df_processed.columns and 'VIP_Missing' in df_processed.columns:
        df_processed['CryoSleep_VIP_Missing'] = df_processed['CryoSleep_Missing'] * df_processed['VIP_Missing']
    
    return df_processed

# Convert categorical variables to numerical with improved encoding
def encode_categorical_features(df):
    df_encoded = df.copy()
    
    # Define categorical columns and their encoding strategies
    categorical_cols = {
        'HomePlanet': 'label',  # Use label encoding for home planet
        'CryoSleep': 'label',  # Use label encoding for cryosleep
        'Deck': 'label',  # Use label encoding for deck
        'Side': 'label',  # Use label encoding for side
        'Destination': 'label',  # Use label encoding for destination
        'VIP': 'label',  # Use label encoding for VIP
        'AgeGroup': 'label',  # Use label encoding for age group
        'Location_Type': 'label',  # Use label encoding for location type
        'CryoSleep_Location_Type': 'label',  # Use label encoding for cryosleep location
        'Spending_Pattern': 'label'  # Use label encoding for spending pattern
    }
    
    # Apply encoding strategies
    for col, strategy in categorical_cols.items():
        if col in df_encoded.columns:
            if strategy == 'label':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    return df_encoded

# Apply preprocessing
df = enhanced_feature_engineering(df)
df = handle_missing_values(df)
df = encode_categorical_features(df)

# Separate features and target
X = df.drop('Transported', axis=1)
y = df['Transported']

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 
                 'CabinNum', 'TotalSpending', 'SpendingPerPerson', 'GroupSize',
                 'DeckLevel', 'LuxuryScore', 'CabinSide', 'CabinDeck',
                 'CryoSleep_Deck', 'CryoSleep_Side', 'VIP_Deck', 'VIP_Side',
                 'Family_Deck', 'Family_Side']

# Add spending ratio columns
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
numerical_cols.extend([f'{col}Ratio' for col in spending_cols])

# Only scale columns that exist and are numerical
numerical_cols = [col for col in numerical_cols if col in X_train.columns]
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])

# Convert boolean target to int
y_train = y_train.astype(int)
y_val = y_val.astype(int)

# Define models with parameter distributions for random search
models = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, verbose=1),
        'params': {
            'n_estimators': randint(100, 500),
            'max_depth': [None] + list(randint(5, 50).rvs(5)),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(
            random_state=42,
            enable_categorical=True,
            early_stopping_rounds=10,
            eval_metric='error',
            verbosity=1
        ),
        'params': {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4)
        }
    }
}

# Perform random search and cross-validation
best_models = {}
crossval_scores = {}
print("\nModel Performance with Random Search:")
for name, model_info in models.items():
    print(f"\nTraining {name}...")
    if name in ['XGBoost', 'LightGBM']:
        random_search = RandomizedSearchCV(
            estimator=model_info['model'],
            param_distributions=model_info['params'],
            n_iter=20,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=2
        )
        random_search.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)]
        )
    else:
        random_search = RandomizedSearchCV(
            estimator=model_info['model'],
            param_distributions=model_info['params'],
            n_iter=20,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=2
        )
        random_search.fit(X_train, y_train)
    best_models[name] = random_search.best_estimator_
    crossval_scores[name] = random_search.best_score_
    print(f"\n{name}:")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.3f}")
    print(f"Validation score: {random_search.best_estimator_.score(X_val, y_val):.3f}")

# Create and train a simple neural network with early stopping
def create_nn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

# Train neural network with early stopping
nn_model = create_nn_model(X_train.shape[1])
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history = nn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,  # Increased max epochs since we have early stopping
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate neural network
nn_val_score = nn_model.evaluate(X_val, y_val, verbose=0)[1]
print(f"\nNeural Network Validation Score: {nn_val_score:.3f}")

# Select best model based on validation score
best_model_name = max(best_models.keys(), 
                     key=lambda x: best_models[x].score(X_val, y_val))
best_model = best_models[best_model_name]
print(f"\nBest performing model: {best_model_name}")

# Load and preprocess test data
test_df = pd.read_csv('test.csv')
test_passenger_ids = test_df['PassengerId'].copy()

# Apply the same preprocessing steps to test data
test_df = enhanced_feature_engineering(test_df)
test_df = handle_missing_values(test_df)
test_df = encode_categorical_features(test_df)
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

# Make predictions using the best model
if best_model_name == 'Neural Network':
    predictions = (nn_model.predict(test_df) > 0.5).astype(int)
else:
    predictions = best_model.predict(test_df).astype(int)

# Create results dataframe with 1/0 values
results_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Transported': predictions
})

# Print prediction summary using 1/0
print("\n" + "="*50)
print("PREDICTION SUMMARY")
print("="*50)
print(f"Total predictions made: {len(predictions)}")
print(f"Number of True predictions (1): {sum(predictions == 1)}")
print(f"Number of False predictions (0): {sum(predictions == 0)}")
print(f"Prediction ratio (1/0): {sum(predictions == 1)/sum(predictions == 0):.2f}")

# Convert to True/False only when saving to CSV
results_df['Transported'] = results_df['Transported'].map({1: 'True', 0: 'False'})

# Save results to CSV
results_df.to_csv('results.csv', index=False)
print("\nResults have been saved to results.csv")

# At the end, after the summary section, print cross-validation accuracy for each model
print("\n" + "="*50)
print("CROSS-VALIDATION ACCURACY SCORES")
print("="*50)
for name, score in crossval_scores.items():
    print(f"{name}: {score:.4f}")
