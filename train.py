from sklearn.tree import DecisionTreeRegressor
from misc import load_data, preprocess_data, train_model, evaluate_model

def main():
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Initialize and train model
    model = DecisionTreeRegressor(random_state=42)
    trained_model = train_model(model, X_train, y_train)
    
    # Evaluate model
    mse = evaluate_model(trained_model, X_test, y_test)
    print(f"Decision Tree MSE: {mse:.4f}")
    
    return mse

if __name__ == "__main__":
    main()