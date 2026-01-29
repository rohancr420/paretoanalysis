import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plotting style
plt.style.use('default')

# Load data
df = pd.read_csv("output.csv")
filtered_data = df[df['Top20Percent'].astype(str).str.lower() == 'true']

print(f" Analyzing {len(filtered_data)} records with Top20Percent = 'true'")
print(f"Available columns: {list(filtered_data.columns)}")

# getting columns
numeric_cols = filtered_data.select_dtypes(include=['number']).columns.tolist()
categorical_cols = filtered_data.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

# subplots
n_numeric = len(numeric_cols)
n_categorical = min(2, len(categorical_cols))  # Limit to 2 categorical plots
total_plots = n_numeric + n_categorical

if total_plots == 0:
    print("No numeric or categorical columns found for visualization.")
else:
    # Create subplots
    n_rows = (total_plots + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    
    # flat axis so its easy to index it
    if total_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot numeric columns
    for i, col in enumerate(numeric_cols):
        if plot_idx < len(axes):
            filtered_data[col].hist(ax=axes[plot_idx], bins=30, edgecolor='black', alpha=0.7)
            axes[plot_idx].set_title(f'Distribution of {col}')
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel('Frequency')
            plot_idx += 1
    
    # code for getting top 10
    for i, col in enumerate(categorical_cols[:2]):  # Limit to first 2 categorical
        if plot_idx < len(axes):
            top_categories = filtered_data[col].value_counts().head(10)
            top_categories.plot(kind='bar', ax=axes[plot_idx], color='skyblue')
            axes[plot_idx].set_title(f'Top 10 {col}')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            plot_idx += 1
    
    # getting rid of empty stuff
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# correlation heatmap
if len(numeric_cols) >= 2:
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = filtered_data[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Columns')
    plt.tight_layout()
    plt.show()

# summary
if numeric_cols:
    print("\n SUMMARY STATISTICS:")
    print(filtered_data[numeric_cols].describe())