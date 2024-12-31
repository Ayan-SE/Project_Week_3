import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encode_categorical_columns(df, columns):
  """
  Performs label encoding on specified categorical columns in a pandas DataFrame.
  """
  df_encoded = df.copy() 
  for col in columns:
    # Check the data type of the column
    if df[col].dtype == 'object':  # Categorical column
      le = LabelEncoder()
      df_encoded[col] = le.fit_transform(df[col]) 
  return df_encoded