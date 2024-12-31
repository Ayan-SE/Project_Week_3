from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def build_decision_tree_model(X, y, test_size=0.2, random_state=42, max_depth=None):
  """
  Builds a decision tree model and evaluates its accuracy.

  """

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=random_state
  )

  # Create a Decision Tree Classifier
  model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

  # Train the model
  model.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred = model.predict(X_test)

  # Calculate accuracy
  accuracy = accuracy_score(y_test, y_pred)

  return model, accuracy

def train_random_forest(X, y,test_size=0.2, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=random_state
     )

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model
