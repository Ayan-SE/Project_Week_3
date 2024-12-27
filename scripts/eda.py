# import libraries
import pandas as pd
import numpy as np

# To calculate  the variability for numerical features 
def calculate_variability(data, features):
     """
      Calculates the variability of numerical features in a DataFrame.
     """
     variability= pd.DataFrame(index=features)
     variability['variance'] = data[features].var()
     variability['std_dev'] = data[features].std()
     variability['coeff_variation'] = data[features].std() / data[features].mean() 

     return variability
   

def check_missing_values(data):
  """
  Checks for missing values in a pandas DataFrame and provides a summary.
  """
  missing_values = pd.DataFrame(index=data.columns)
  missing_values['Missing Count'] = data.isnull().sum()
  missing_values['Missing Percentage'] = (data.isnull().sum() / len(data)) * 100

  return missing_values

def compare_changes(data, id_col, time_col, columns_to_compare):
    """
    Compare changes in specified columns grouped by ID over time.

    """
    # Sort data by ID and time
    data = data.sort_values(by=[id_col, time_col])

    # Compute differences for the specified columns
    changes = data.groupby(id_col)[columns_to_compare].apply(lambda x: x.diff())

    # Merge changes back to the original DataFrame for context
    result = data.copy()
    for col in columns_to_compare:
        result[f"Change_in_{col}"] = changes[col]

    return result

#DataAnalysis()