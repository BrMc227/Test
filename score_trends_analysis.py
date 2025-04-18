import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind  

# variables
BOY_FILE = 'quality_takehome_BOY_file.csv'
MOY_FILE = 'quality_takehome_MOY_file.csv'
OUTPUT_DIR = 'arm_comparison_plots' # Directory to save plots
DIAGNOSTIC_SCORES = ['Adjusted WCPM Score', 'PA PR', 'SR PR', 'Vocabulary PR', 'Decoding PR']
ARM_SCORE_COL = 'ARM' 
GRADE_ORDER = ['Kindergarten', 'First Grade', 'Second Grade', 'Third Grade', 'Fourth Grade', 'Fifth Grade']

# Color Scheme
COLOR_COMPLETE = '#20c997' # Teal for Complete Data classes
COLOR_PARTIAL = '#ff9800'  # Orange for Partial
COLOR_HIGH_MISSING = '#6f42c1' # Purple for High Missing classes

# functions

def load_data(boy_file, moy_file):
    """Loads, combines, and prepares BOY and MOY data."""
    if not os.path.exists(boy_file) or not os.path.exists(moy_file):
        print(f"Error: Data files not found. Ensure '{boy_file}' and '{moy_file}' are present.")
        return None
    try:
        boy_df = pd.read_csv(boy_file)
        boy_df['Form'] = 'BOY'
        moy_df = pd.read_csv(moy_file)
        moy_df['Form'] = 'MOY'
        df = pd.concat([boy_df, moy_df], axis=0, ignore_index=True)
        print(f"Data loaded successfully. Total rows: {len(df)}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_scores(df, score_columns):
    """Converts score columns to numeric, coercing errors."""
    print("Cleaning score columns...")
    if df is None: return None
    cleaned_df = df.copy()
    valid_cols_found = []
    for col in score_columns:
        if col in cleaned_df.columns:
            valid_cols_found.append(col)
            original_type = cleaned_df[col].dtype
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            if cleaned_df[col].isnull().any() and not pd.api.types.is_numeric_dtype(original_type):
                print(f"  - Converted '{col}' to numeric; some values became NaN.")
        else:
            print(f"  - Warning: Score column '{col}' not found.")
    if not valid_cols_found:
        print("Error: No valid score columns found to clean.")
        return None
    return cleaned_df

def categorize_class_completeness(df, diagnostic_scores):
    """Calculates % students missing any diagnostic and categorizes classes."""
    print("Categorizing classes by diagnostic data completeness...")
    if df is None or df.empty: return None

    # Ensure necessary diagnostic scores and grouping columns exist
    required_group_cols = ['schoolId', 'Grade', 'classId']
    valid_diagnostic_scores = [col for col in diagnostic_scores if col in df.columns]

    if not valid_diagnostic_scores:
        print("Error: No diagnostic score columns found for completeness check.")
        return None
    if not all(col in df.columns for col in required_group_cols):
        missing_group = [col for col in required_group_cols if col not in df.columns]
        print(f"Error: Missing required grouping columns for categorization: {', '.join(missing_group)}")
        return None

    temp_df = df.copy()
    # Check if *any* valid diagnostic is missing for each student row
    temp_df['Missing_Any_Diagnostic'] = temp_df[valid_diagnostic_scores].isnull().any(axis=1)

    # Group by class to find percentage missing
    class_group_cols = ['schoolId', 'Grade', 'classId']
    status_summary = temp_df.groupby(class_group_cols).agg(
        Total_Students=('classId', 'size'),
        Students_Missing_Any=('Missing_Any_Diagnostic', 'sum')
    ).reset_index()
    status_summary['Diagnostic_Missing_Percent'] = status_summary.apply(
        lambda row: (row['Students_Missing_Any'] / row['Total_Students'] * 100) if row['Total_Students'] > 0 else 0, axis=1
    )


    # Categorize based on the percentage
    def categorize_status(percent):
        if pd.isna(percent): return 'Unknown'
        if percent > 50.0: return 'High Missing (>50%)'
        if percent == 0.0: return 'Complete (0%)'
        return 'Partial (1-50%)'

    status_summary['Completeness_Status'] = status_summary['Diagnostic_Missing_Percent'].apply(categorize_status)
    print(f"Found {len(status_summary[status_summary['Completeness_Status'] == 'Complete (0%)'])} classes with complete data.")
    print(f"Found {len(status_summary[status_summary['Completeness_Status'] == 'High Missing (>50%)'])} classes with high missing data.")
    print(f"Found {len(status_summary[status_summary['Completeness_Status'] == 'Partial (1-50%)'])} classes with partial missing data.")

    return status_summary[class_group_cols + ['Completeness_Status']]

def calculate_class_arm_trends(df, arm_score_col):
    """Calculates BOY Avg, MOY Avg, and Change in ARM score per class."""
    print("Calculating ARM score trends per class...")
    if df is None or df.empty:
         print(f"Skipping ARM trend calculation: no data.")
         return None
    if arm_score_col not in df.columns:
        print(f"Skipping ARM trend calculation: ARM column '{arm_score_col}' not found.")
        return None

    class_group_cols = ['schoolId', 'Grade', 'classId']
    required_cols = class_group_cols + ['Form', arm_score_col]
    if not all(col in df.columns for col in required_cols):
         missing = [col for col in required_cols if col not in df.columns]
         print(f"Skipping ARM trend calculation: missing columns {missing}.")
         return None

    # Calculate mean, handling potential non-numeric grouping cols if any
    try:
         arm_trends = df.groupby(class_group_cols + ['Form'])[arm_score_col].mean(numeric_only=True).unstack('Form')
    except Exception as e:
         print(f"Error during ARM trend grouping/mean calculation: {e}")
         return None

    arm_trends.columns = [f'{arm_score_col}_{form}' for form in arm_trends.columns] # e.g., Adjusted WCPM Score_BOY

    boy_col = f'{arm_score_col}_BOY'
    moy_col = f'{arm_score_col}_MOY'

    # Ensure columns exist before calculating change
    arm_trends[f'{arm_score_col}_Change'] = np.nan # Initialize change column
    boy_exists = boy_col in arm_trends.columns
    moy_exists = moy_col in arm_trends.columns

    if boy_exists and moy_exists:
        arm_trends[f'{arm_score_col}_Change'] = arm_trends[moy_col] - arm_trends[boy_col]
    else:
        missing_forms = []
        if not boy_exists: missing_forms.append("BOY")
        if not moy_exists: missing_forms.append("MOY")
        print(f"  - Warning: Could not calculate ARM change for some classes, missing {', '.join(missing_forms)} data.")


    return arm_trends.reset_index()

def analyze_trends_by_completeness(class_status, arm_trends, arm_score_col):
    """Merges status and trends, aggregates by grade/status, and prints results."""
    print("\n--- Comparing ARM Trends by Diagnostic Completeness Status ---")
    if class_status is None or arm_trends is None:
        print("Cannot perform comparison due to missing status or trend data.")
        return None

    # Merge status and ARM trends
    merged_data = pd.merge(class_status, arm_trends, on=['schoolId', 'Grade', 'classId'], how='inner')

    # --- Keep ALL statuses for aggregation ---
    comparison_groups = merged_data.copy() # Use all merged data

    # Aggregate results by Grade and Status
    boy_col = f'{arm_score_col}_BOY'
    moy_col = f'{arm_score_col}_MOY'
    change_col = f'{arm_score_col}_Change'

    agg_dict = {'Num_Classes': ('schoolId', 'size')}
    if boy_col in comparison_groups.columns: agg_dict['Avg_BOY_ARM'] = (boy_col, 'mean')
    if moy_col in comparison_groups.columns: agg_dict['Avg_MOY_ARM'] = (moy_col, 'mean')
    if change_col in comparison_groups.columns: agg_dict['Avg_ARM_Change'] = (change_col, 'mean')

    if len(agg_dict) <= 1:
         print("Error: Missing required ARM score columns for aggregation.")
         return None

    try:
        final_comparison = comparison_groups.groupby(['Grade', 'Completeness_Status']).agg(**agg_dict)
        print("\nComparison Results (Avg Scores & Change by Grade/Status):")
        # Print the unstacked version for easy reading in console
        print(final_comparison.unstack('Completeness_Status').to_string(float_format='%.1f'))
        # Return the *reset* version which is easier for the plotting function
        return final_comparison.reset_index()
    except Exception as e:
        print(f"Error during final aggregation/comparison: {e}")
        return None

def run_ttests_by_group(df, arm_score_col):
    """Runs t-tests comparing ARM scores between completeness groups."""
    print("\n--- Running t-tests between Class Completeness Groups ---")
    if df is None or df.empty:
        print("Skipping t-tests: no data.")
        return

    boy_col = f'{arm_score_col}_BOY'
    moy_col = f'{arm_score_col}_MOY'
    change_col = f'{arm_score_col}_Change'
    
    score_cols = [boy_col, moy_col, change_col]
    statuses = ['Complete (0%)', 'Partial (1-50%)', 'High Missing (>50%)']

    for score in score_cols:
        if score not in df.columns:
            continue
        print(f"\nT-Tests for {score}:")
        for i in range(len(statuses)):
            for j in range(i+1, len(statuses)):
                group1 = df[df['Completeness_Status'] == statuses[i]][score].dropna()
                group2 = df[df['Completeness_Status'] == statuses[j]][score].dropna()
                if len(group1) > 1 and len(group2) > 1:
                    stat, p = ttest_ind(group1, group2, equal_var=False)  # Welch's t-test
                    print(f"  {statuses[i]} vs {statuses[j]}: p = {p:.4f}")
                else:
                    print(f"  Not enough data for {statuses[i]} vs {statuses[j]}.")


def plot_comparison(comparison_data, value_col, y_label, title, output_file, color_complete, color_partial, color_high_missing, grade_order):
    """Creates a grouped bar chart for a specific comparison metric, including Partial."""
    print(f"Generating plot: {title}...")
    if comparison_data is None or comparison_data.empty:
        print(f"Skipping plot '{title}': No comparison data.")
        return
    if value_col not in comparison_data.columns:
        print(f"Skipping plot '{title}': Missing value column '{value_col}'.")
        return

    # Pivot data for plotting
    try:
        plot_pivot = comparison_data.pivot(index='Grade', columns='Completeness_Status', values=value_col)
    except Exception as e:
        print(f"Error pivoting data for plot '{title}': {e}")
        return

    # Include all three statuses if they exist after pivot
    cols_to_plot = ['Complete (0%)', 'Partial (1-50%)', 'High Missing (>50%)']
    valid_cols_to_plot = [col for col in cols_to_plot if col in plot_pivot.columns] # Ensure columns exist
    plot_pivot = plot_pivot[valid_cols_to_plot] # Select only existing columns
    plot_pivot = plot_pivot.reindex(grade_order).dropna(how='all')

    if plot_pivot.empty or len(plot_pivot.columns) == 0: # Check if any columns remain
        print(f"Skipping plot '{title}': No data left after reindexing/filtering for specified statuses.")
        return

    # Plotting setup
    fig, ax = plt.subplots(figsize=(14, 7))
    grades = plot_pivot.index
    n_grades = len(grades)
    n_groups = len(plot_pivot.columns)
    x = np.arange(n_grades)
    width = 0.8 / n_groups # Adjust width based on number of groups plotted

    # Map status to color
    color_map = {
        'Complete (0%)': color_complete,
        'Partial (1-50%)': color_partial,
        'High Missing (>50%)': color_high_missing
    }

    # Calculate offsets for bars
    offsets = np.linspace(-width * (n_groups - 1) / 2, width * (n_groups - 1) / 2, n_groups)

    rects = {}
    for i, status in enumerate(plot_pivot.columns): # Iterate through available columns
        offset = offsets[i]
        color = color_map.get(status, 'gray') # Default to gray if status not in map
        rect = ax.bar(x + offset, plot_pivot[status].fillna(0), width, label=status, color=color)
        rects[status] = rect # Store rects by status name

    # Customize
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(grades)
    ax.legend(title="Class Status")
    ax.grid(False)
    ax.axhline(0, color='grey', linewidth=0.8) # Add horizontal line at 0

    # Add labels
    for status, rect_group in rects.items(): # Iterate through stored rects
         ax.bar_label(rect_group, padding=3, fmt='%.1f')

    ax.margins(y=0.1)
    fig.tight_layout()

    # Save
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        full_path = os.path.join(OUTPUT_DIR, output_file)
        plt.savefig(full_path, dpi=300)
        print(f"Plot saved to '{full_path}'")
    except Exception as e:
        print(f"Error saving plot '{output_file}': {e}")
    finally:
        plt.close(fig)


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting ARM Score Comparison by Data Completeness ---")

    # 1. Load Data
    df_combined = load_data(BOY_FILE, MOY_FILE)

    # 2. Clean Score Columns
    all_scores_to_clean = list(set(DIAGNOSTIC_SCORES + [ARM_SCORE_COL]))
    df_cleaned = clean_scores(df_combined, all_scores_to_clean)

    # 3. Categorize Classes by Completeness
    class_status_df = categorize_class_completeness(df_cleaned, DIAGNOSTIC_SCORES)

    # 4. Calculate Class ARM Trends (BOY, MOY, Change)
    class_arm_trends_df = calculate_class_arm_trends(df_cleaned, ARM_SCORE_COL)

    # 5. Analyze Trends by Completeness Status (Merge, Aggregate, Print Table)
    comparison_results = analyze_trends_by_completeness(class_status_df, class_arm_trends_df, ARM_SCORE_COL)

    # 6. Generate Visualizations
    if comparison_results is not None:
        # Plot Avg BOY ARM Score
        plot_comparison(comparison_results, value_col='Avg_BOY_ARM', y_label='Average BOY ARM Score',
                        title='Average BOY ARM Score by Class Data Completeness',
                        output_file='comparison_avg_boy_arm.png',
                        color_complete=COLOR_COMPLETE, color_partial=COLOR_PARTIAL, color_high_missing=COLOR_HIGH_MISSING, grade_order=GRADE_ORDER)

        # Plot Avg MOY ARM Score
        plot_comparison(comparison_results, value_col='Avg_MOY_ARM', y_label='Average MOY ARM Score',
                        title='Average MOY ARM Score by Class Data Completeness',
                        output_file='comparison_avg_moy_arm.png',
                        color_complete=COLOR_COMPLETE, color_partial=COLOR_PARTIAL, color_high_missing=COLOR_HIGH_MISSING, grade_order=GRADE_ORDER)

        # Plot Avg Change in ARM Score
        plot_comparison(comparison_results, value_col='Avg_ARM_Change', y_label='Average Change in ARM Score (MOY - BOY)',
                        title='Average Change in ARM Score by Class Data Completeness',
                        output_file='comparison_avg_arm_change.png',
                        color_complete=COLOR_COMPLETE, color_partial=COLOR_PARTIAL, color_high_missing=COLOR_HIGH_MISSING, grade_order=GRADE_ORDER)

    print("\n--- Analysis Complete ---")
