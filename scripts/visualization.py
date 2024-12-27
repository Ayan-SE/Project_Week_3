import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_histograms(data, num_cols, bins=10, figsize=(4, 2)):
     """
     Plots histograms for specified numerical columns in a DataFrame.
    """
     for col in num_cols:
       plt.figure(figsize=figsize)
       plt.hist(data[col], bins=bins)
       plt.xlabel(col)
       plt.ylabel('Frequency')
       plt.title(f'Histogram of {col}')
       plt.show()

def plot_bar_charts(df, cat_cols, figsize=(10, 5)):
  """
  Plots bar charts for specified categorical columns in a DataFrame.
  """
  for col in cat_cols:
    # Check if the column is categorical
    if df[col].dtype == 'object' or df[col].dtype == 'category': 
      plt.figure(figsize=figsize)
      df[col].value_counts().plot(kind='bar')
      plt.xlabel(col)
      plt.ylabel('Count')
      plt.title(f'Bar Chart of {col}')
      plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
      plt.show()
    else:
      print(f"Column '{col}' is not categorical. Skipping bar chart.")


# Function to create scatter plots
def plot_relationships(data, x_col, y_col, hue_col):
      """
       Create scatter plots to explore relationships between two columns as a function of a third column.
     """
      plt.figure(figsize=(8, 6))
      unique_hues = data[hue_col].unique()
      colors = plt.cm.viridis(range(len(unique_hues)))
      color_map = dict(zip(unique_hues, colors))

      for hue_value in unique_hues:
        subset = data[data[hue_col] == hue_value]
        plt.scatter(subset[x_col], subset[y_col], label=hue_value, color=color_map[hue_value])
        plt.title(f"Scatter Plot of {y_col} vs {x_col} by {hue_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


# Function to compute and visualize correlation matrices
def plot_correlation_matrix(data, group_col, value_cols):
    """
    Compute and visualize correlation matrices for groups defined by a column.
    """
    if group_col not in data.columns:
        for group, group_data in data.groupby(group_col):
           print(f"Correlation matrix for {group_col}: {group}")
           correlation_matrix = group_data[value_cols].corr()
           print(correlation_matrix)
  
        # Plot heatmap
        plt.figure(figsize=(8, 6))
       # sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title(f"Correlation Matrix for {group_col}: {group}")
        plt.show()
    else:
        print(f"Column '{group_col}' not found in data.")
    

# Function to calculate changes over time
def calculate_changes(df, group_by_columns, value_column):
    """
    Calculates the percentage change for a given value column, grouped by specific columns.
    """
    grouped = df.groupby(group_by_columns)[value_column].sum().unstack()
    changes = grouped.pct_change(axis=1).stack().reset_index()
    changes.columns = group_by_columns + [f"{value_column}_change"]
    return changes

# Function to plot trends over geography
def plot_trends(df, geography_col, trend_col, change_col, title):
    """
    Plots trends in percentage change over geography.
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=geography_col, y=change_col, hue=trend_col, marker="o")
    plt.title(title)
    plt.xlabel(geography_col)
    plt.ylabel("Percentage Change")
    plt.legend(title=trend_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
def detect_outliers(data, numerical_columns):
    """
    Detect outliers in numerical columns using box plots.
    """
    outliers = {}

    for col in numerical_columns:
        # Compute Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers[col] = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]

        # Plot box plot
        plt.figure(figsize=(8, 6))
        plt.boxplot(data[col], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        plt.title(f"Box Plot for {col}")
        plt.xlabel(col)
        plt.show()

    return outliers

