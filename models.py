import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Load the training data
data = pd.read_csv('train.csv')

# Display the first few rows of the dataframe
print(data.head())

# Separate features and target variable
X = data.drop(columns=['Id', 'Target'])
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def RandomForest():
    # Initialize the model
    model = RandomForestClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Load the test data
    test_data = pd.read_csv('test.csv')

    # Preprocess the test data (drop the Id column and standardize the features)
    X_test_data = test_data.drop(columns=['Id'])
    X_test_data = scaler.transform(X_test_data)

    # Make predictions on the test data
    test_predictions = model.predict(X_test_data)

    # Create a new DataFrame with only the Id and Target columns
    output = pd.DataFrame({
        'Id': test_data['Id'],
        'Target': test_predictions
    })

    return (output, accuracy)


def AdaBoost():
    # Define the parameter grid for AdaBoost
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    }

    # Initialize the AdaBoost model
    model = AdaBoostClassifier(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

    # Train the model using GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Make predictions on the validation set
    y_val_pred = best_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_val_pred)

    # Print the best parameters and best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Load the test data
    test_data = pd.read_csv('test.csv')

    # Preprocess the test data (drop the Id column and standardize the features)
    X_test_data = test_data.drop(columns=['Id'])
    X_test_data = scaler.transform(X_test_data)

    # Make predictions on the test data
    test_predictions = best_model.predict(X_test_data)

    # Create a new DataFrame with only the Id and Target columns
    output = pd.DataFrame({
        'Id': test_data['Id'],
        'Target': test_predictions
    })

    return (output, accuracy)


def DecisionTree():
    # Initialize the Decision Tree model
    tree = DecisionTreeClassifier(random_state=42)

    # Train the model
    tree.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = tree.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Load the test data
    test_data = pd.read_csv('test.csv')

    # Preprocess the test data (drop the Id column and standardize the features)
    X_test_data = test_data.drop(columns=['Id'])
    X_test_data = scaler.transform(X_test_data)

    # Make predictions on the test data
    test_predictions = tree.predict(X_test_data)

    # Create a new DataFrame with only the Id and Target columns
    output = pd.DataFrame({
        'Id': test_data['Id'],
        'Target': test_predictions
    })

    return (output, accuracy)



def GradientBoosting():
    # Initialize the model
    model = GradientBoostingClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Load the test data
    test_data = pd.read_csv('test.csv')

    # Preprocess the test data (drop the Id column and standardize the features)
    X_test_data = test_data.drop(columns=['Id'])
    X_test_data = scaler.transform(X_test_data)

    # Make predictions on the test data
    test_predictions = model.predict(X_test_data)

    # Create a new DataFrame with only the Id and Target columns
    output = pd.DataFrame({
        'Id': test_data['Id'],
        'Target': test_predictions
    })

    return (output, accuracy)



def SupportVectorMachine():
    # Initialize the model with a linear kernel for faster computation
    model = SVC(kernel='linear', random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Load the test data
    test_data = pd.read_csv('test.csv')

    # Preprocess the test data (drop the Id column and standardize the features)
    X_test_data = test_data.drop(columns=['Id'])
    X_test_data = scaler.transform(X_test_data)

    # Make predictions on the test data
    test_predictions = model.predict(X_test_data)

    # Create a new DataFrame with only the Id and Target columns
    output = pd.DataFrame({
        'Id': test_data['Id'],
        'Target': test_predictions
    })

    return (output, accuracy)

def KNearestNeighbors():
    from sklearn.neighbors import KNeighborsClassifier
    # Initialize the KNN model
    model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Load the test data
    test_data = pd.read_csv('test.csv')

    # Preprocess the test data (drop the Id column and standardize the features)
    X_test_data = test_data.drop(columns=['Id'])
    X_test_data = scaler.transform(X_test_data)

    # Make predictions on the test data
    test_predictions = model.predict(X_test_data)

    # Create a new DataFrame with only the Id and Target columns
    output = pd.DataFrame({
        'Id': test_data['Id'],
        'Target': test_predictions
    })

    return (output, accuracy)
