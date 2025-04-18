#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# variables
BOY_FILE = 'quality_takehome_BOY_file.csv'
MOY_FILE = 'quality_takehome_MOY_file.csv'
OUTPUT_PLOT_FILE = 'missing_diagnostics_by_grade_form.png'
DIAGNOSTIC_SCORES = ['Adjusted WCPM Score', 'PA PR', 'SR PR', 'Vocabulary PR', 'Decoding PR']
GRADE_ORDER = ['Kindergarten', 'First Grade', 'Second Grade', 'Third Grade', 'Fourth Grade', 'Fifth Grade']
MISSING_THRESHOLD = 50.0 # Percentage threshold for "high missing"

# Read and transform BOY file
boy_df = pd.read_csv('quality_takehome_BOY_file.csv')
boy_df['Form'] = 'BOY'


boy_df.head()


# Read and transform MOY file
moy_df = pd.read_csv('quality_takehome_MOY_file.csv')
moy_df['Form'] = 'MOY'


moy_df.head()


#combine datasets; first check headers match

print("\nColumns match:", set(boy_df.columns) == set(moy_df.columns))


#combine datasets
Amira_EDA = pd.concat([boy_df, moy_df], axis=0, ignore_index=True)


Amira_EDA.head()


#one more check we've concatenated properly
Amira_EDA['Form'].value_counts()


Amira_EDA.info()


Amira_EDA.describe()


#import sweetviz for eda report
import sweetviz as sv


#compare dataset values based on diff BOY/MOY for any overt questions or areas to dig in

comparison_report = sv.compare(
    [boy_df, "BOY Data"],
    [moy_df, "MOY Data"],
    pairwise_analysis='off'
)

comparison_report.show_html('Amira_BOY_MOY_Comparison.html',
                          open_browser=True,
                          layout='widescreen')


# Define component scores at the beginning
component_scores = ['Adjusted WCPM Score', 'PA PR', 'SR PR', 'Vocabulary PR', 'Decoding PR']

# Print the counts of BOY and MOY tests
print("Number of tests by Form:")
print(Amira_EDA['Form'].value_counts())

# Print percentages of missing tests by Form
print("\nPercentage of missing tests by Form:")
for form in ['BOY', 'MOY']:
    total_tests = len(Amira_EDA[Amira_EDA['Form']==form])
    for component in component_scores:
        missing = Amira_EDA[Amira_EDA['Form']==form][component].isnull().sum()
        percent = (missing/total_tests*100).round(1)
        print(f"{form} - {component}: {missing}/{total_tests} ({percent}%)")

# Visualize missing component tests by Form

# Create figure with larger size
plt.figure(figsize=(15, 8))

# Calculate missing counts by Form
boy_missing = Amira_EDA[Amira_EDA['Form']=='BOY'][component_scores].isnull().sum()
moy_missing = Amira_EDA[Amira_EDA['Form']=='MOY'][component_scores].isnull().sum()

# Get totals for percentage calculation
boy_total = len(Amira_EDA[Amira_EDA['Form']=='BOY'])
moy_total = len(Amira_EDA[Amira_EDA['Form']=='MOY'])

# Set up the bar positions
x = np.arange(len(component_scores))
width = 0.35

# Use seaborn's default color palette
sns.set_palette("deep")
colors = sns.color_palette()

# Create bars with seaborn's default colors
boy_bars = plt.bar(x - width/2, boy_missing, width, label=f'BOY (n={boy_total})', color=colors[0])
moy_bars = plt.bar(x + width/2, moy_missing, width, label=f'MOY (n={moy_total})', color=colors[1])

# Add percentage labels for BOY
for bar in boy_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}\n({(height/boy_total*100):.1f}%)',
             ha='center', va='bottom')

# Add percentage labels for MOY
for bar in moy_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}\n({(height/moy_total*100):.1f}%)',
             ha='center', va='bottom')

# Customize the plot
plt.title('Missing Component Tests Distribution by Form')
plt.xlabel('Component Test')
plt.ylabel('Number of Missing Tests')
plt.xticks(x, component_scores, rotation=45, ha='right')

# Move legend to bottom right
plt.legend(loc='lower right')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save and show the plot
plt.savefig('missing_components_by_form.png', dpi=300, bbox_inches='tight')
plt.show()

# Check student completion by grade
# Define the correct grade order
grade_order = ['Kindergarten', 'First Grade', 'Second Grade', 
               'Third Grade', 'Fourth Grade', 'Fifth Grade']

# Create the grade level plot with ordered categories
plt.figure(figsize=(12, 6))
sns.countplot(data=Amira_EDA, x='Grade', hue='Form', order=grade_order)
plt.title('Distribution of Assessments by Grade and Form')
plt.xticks(rotation=45)
plt.show()


# 4. Missing Data Patterns by Form
missing_by_form = Amira_EDA.groupby('Form').apply(lambda x: x.isnull().sum()).unstack()
print("\nMissing Data by Form:")
print(missing_by_form)


#digging into why so many nulls for diagnostic percentile ranks 

# Is student is missing diagnostics
Amira_EDA['Has_Missing_Component_Scores'] = Amira_EDA[component_scores].isnull().any(axis=1)


#How many are they missing 
Amira_EDA['Total_Missing_Component_Scores'] = Amira_EDA[component_scores].isnull().sum(axis=1)


Amira_EDA.head()


# Display distribution of missing diagnostics
print("\nDistribution of total missing diagnostics per student:")
print(Amira_EDA['Total_Missing_Component_Scores'].value_counts().sort_index())



missing_report = Amira_EDA.groupby(['schoolId', 'classId', 'Form'], observed=True).agg({
    'Total_Missing_Component_Scores': 'sum',  # Total number of missing diagnostics
    'Has_Missing_Component_Scores': ['sum', 'size']  # sum = number of students with missing, size = total students
})


missing_report['Percent_Students_Missing'] = (missing_report[('Has_Missing_Component_Scores', 'sum')] / 
                                            missing_report[('Has_Missing_Component_Scores', 'size')] * 100)


# Rename columns for clarity
missing_report.columns = ['Total_Missing_Component_Scores', 'Students_With_Missing', 'Total_Students', 'Percent_Missing']


missing_report_sorted = missing_report.sort_values('Total_Missing_Component_Scores', ascending=False)



print("\nComprehensive Missing Component Scores Report:")
print(missing_report_sorted.head(20))


#group by school,class, student, form
Amira_EDA = Amira_EDA.groupby(['schoolId', 'classId', 'Student ID', 'Form'])


#check grouping
Amira_EDA.first().head()


# Load the data
boy_df = pd.read_csv('quality_takehome_BOY_file.csv')
boy_df['Form'] = 'BOY'
moy_df = pd.read_csv('quality_takehome_MOY_file.csv')
moy_df['Form'] = 'MOY'

# Combine the data
df = pd.concat([boy_df, moy_df], axis=0, ignore_index=True)

# Define diagnostic scores
diagnostic_scores = ['Adjusted WCPM Score', 'PA PR', 'SR PR', 'Vocabulary PR', 'Decoding PR']

# Identify students with any missing diagnostic test
df['Has_Missing_Diagnostics'] = df[diagnostic_scores].isnull().any(axis=1)

# --- School Level Analysis (for console output) ---
school_summary = df.groupby(['schoolId', 'Form']).agg(
    Total_Students=('schoolId', 'count'),
    Students_Missing_Tests=('Has_Missing_Diagnostics', 'sum')
).reset_index()
school_summary['Missing_Student_%'] = (school_summary['Students_Missing_Tests'] / school_summary['Total_Students'] * 100).round(1)
high_missing_schools = school_summary[school_summary['Missing_Student_%'] > 50].sort_values('Missing_Student_%', ascending=False)
perfect_missing_schools = school_summary[school_summary['Missing_Student_%'] == 100.0]
high_missing_schools_display = high_missing_schools.rename(columns={
    'schoolId': 'School ID', 'Students_Missing_Tests': 'Students Missing Tests',
    'Total_Students': 'Total Students', 'Missing_Student_%': 'Missing Student %'
})

# --- Class Level Analysis (for console output) ---
class_summary = df.groupby(['schoolId', 'Grade', 'classId', 'Form']).agg(
    Total_Students=('classId', 'count'),
    Students_Missing_Tests=('Has_Missing_Diagnostics', 'sum')
).reset_index()
class_summary['Missing_Student_%'] = (class_summary['Students_Missing_Tests'] / class_summary['Total_Students'] * 100).round(1)
high_missing_classes = class_summary[class_summary['Missing_Student_%'] > 50].sort_values(['schoolId', 'Grade', 'Missing_Student_%'], ascending=[True, True, False])
perfect_missing_classes = class_summary[class_summary['Missing_Student_%'] == 100.0]
high_missing_classes_display = high_missing_classes.rename(columns={
    'schoolId': 'School ID', 'classId': 'Class ID', 'Students_Missing_Tests': 'Students Missing Tests',
    'Total_Students': 'Total Students', 'Missing_Student_%': 'Missing Student %'
})

# --- Visualization: Missing Diagnostics by Grade and Form ---

# Calculate counts of students with missing diagnostics per Grade and Form
missing_by_grade_form = df[df['Has_Missing_Diagnostics']].groupby(['Grade', 'Form']).size().unstack(fill_value=0)

# Ensure both BOY and MOY columns exist, even if one has no missing data for a grade
if 'BOY' not in missing_by_grade_form.columns: missing_by_grade_form['BOY'] = 0
if 'MOY' not in missing_by_grade_form.columns: missing_by_grade_form['MOY'] = 0

# --- Define specific Grade order ---
# Note: Ensure these strings exactly match the values in your 'Grade' column
grade_order = ['Kindergarten', 'First Grade', 'Second Grade', 'Third Grade', 'Fourth Grade', 'Fifth Grade']
# Filter and reindex the data to match the desired order
missing_by_grade_form = missing_by_grade_form.reindex(grade_order).dropna(how='all') # dropna removes grades not present in data

# Set up plot
fig, ax = plt.subplots(figsize=(12, 7))
grades = missing_by_grade_form.index # Use the reordered index
x = np.arange(len(grades)) # the label locations
width = 0.35 # the width of the bars

# Colors (same scheme as before)
colors = {'BOY': '#2196F3', 'MOY': '#FF9800'}

# Plot bars
rects1 = ax.bar(x - width/2, missing_by_grade_form['BOY'], width, label='BOY', color=colors['BOY'])
rects2 = ax.bar(x + width/2, missing_by_grade_form['MOY'], width, label='MOY', color=colors['MOY'])

# Add some text for labels, title and axes ticks
ax.set_ylabel('Number of Students with Missing Diagnostics')
ax.set_title('Students with Missing Diagnostics by Grade and Form')
ax.set_xticks(x)
ax.set_xticklabels(grades) # Use the reordered grades as labels
ax.legend(title="Form")

# --- Remove horizontal grid lines ---
# ax.grid(True, axis='y', linestyle='--', alpha=0.7) # This line is now removed/commented out

# Optional: Add vertical grid lines if desired
ax.grid(True, axis='x', linestyle='--', alpha=0.5)


fig.tight_layout()

# Save the plot
plt.savefig('missing_diagnostics_by_grade_form.png', dpi=300)
print("\nVisualization 'missing_diagnostics_by_grade_form.png' saved.")
plt.close(fig) # Close the plot figure to free memory

# --- Output Summary Counts ---
print("\n--- Summary Counts ---")
print("=" * 80)
print(f"Total School Records (>50% Students Missing Any Test): {len(high_missing_schools)}")
print(f"Total School Records (100% Students Missing Any Test): {len(perfect_missing_schools)}")
print(f"Total Class Records (>50% Students Missing Any Test):  {len(high_missing_classes)}")
print(f"Total Class Records (100% Students Missing Any Test): {len(perfect_missing_classes)}")
print("=" * 80)


# --- Output Detailed Tables ---
print("\nSchools with >50% Students Missing At Least One Diagnostic Test")
print("-" * 80)
if not high_missing_schools_display.empty:
    print(high_missing_schools_display[['School ID', 'Form', 'Students Missing Tests', 'Total Students', 'Missing Student %']].to_string(index=False))
else:
    print("No schools found meeting the >50% criteria.")

print("\n\nClasses with >50% Students Missing At Least One Diagnostic Test")
print("-" * 80)
if not high_missing_classes_display.empty:
    print(high_missing_classes_display[['School ID', 'Grade', 'Class ID', 'Form', 'Students Missing Tests', 'Total Students', 'Missing Student %']].to_string(index=False))
else:
    print("No classes found meeting the >50% criteria.")

print("\nAnalysis complete.")

# --- Functions ---

def load_data(boy_file, moy_file):
    """Loads and combines BOY and MOY data."""
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

def calculate_missing_indicators(df, diagnostic_scores):
    """Adds columns indicating if any test is missing."""
    if df is None or df.empty:
        return df
    # Check if columns exist before attempting to check nulls
    valid_scores = [score for score in diagnostic_scores if score in df.columns]
    if not valid_scores:
        print("Warning: None of the specified diagnostic score columns found in data.")
        df['Has_Missing_Diagnostics'] = False # Add column but set to False
    else:
        if len(valid_scores) < len(diagnostic_scores):
             print(f"Warning: Only found columns: {', '.join(valid_scores)}")
        df['Has_Missing_Diagnostics'] = df[valid_scores].isnull().any(axis=1)
    return df

def analyze_missing_rates(df):
    """Calculates missing percentages at School and Class levels."""
    if df is None or df.empty or 'Has_Missing_Diagnostics' not in df.columns:
        print("Skipping analysis due to missing data or prior errors.")
        return None, None

    # School Level
    school_summary = df.groupby(['schoolId', 'Form']).agg(
        Total_Students=('schoolId', 'count'),
        Students_Missing_Tests=('Has_Missing_Diagnostics', 'sum')
    ).reset_index()
    # Avoid division by zero if a group is empty (shouldn't happen with count but good practice)
    school_summary['Missing_Student_%'] = school_summary.apply(
        lambda row: (row['Students_Missing_Tests'] / row['Total_Students'] * 100) if row['Total_Students'] > 0 else 0, axis=1
    ).round(1)


    # Class Level
    class_summary = df.groupby(['schoolId', 'Grade', 'classId', 'Form']).agg(
        Total_Students=('classId', 'count'),
        Students_Missing_Tests=('Has_Missing_Diagnostics', 'sum')
    ).reset_index()
    class_summary['Missing_Student_%'] = class_summary.apply(
         lambda row: (row['Students_Missing_Tests'] / row['Total_Students'] * 100) if row['Total_Students'] > 0 else 0, axis=1
    ).round(1)


    return school_summary, class_summary

def filter_high_missing(school_summary, class_summary, threshold):
    """Filters summaries for high (>threshold) and perfect (100%) missing rates."""
    if school_summary is None or class_summary is None:
        return None, None, None, None

    high_missing_schools = school_summary[school_summary['Missing_Student_%'] > threshold].copy()
    perfect_missing_schools = school_summary[school_summary['Missing_Student_%'] == 100.0].copy()

    high_missing_classes = class_summary[class_summary['Missing_Student_%'] > threshold].copy()
    perfect_missing_classes = class_summary[class_summary['Missing_Student_%'] == 100.0].copy()

    return high_missing_schools, perfect_missing_schools, high_missing_classes, perfect_missing_classes

def print_summary_counts(high_schools, perfect_schools, high_classes, perfect_classes, threshold):
    """Prints the summary counts section."""
    print("\n--- Summary Counts ---")
    print("=" * 80)
    # Check if data is None before accessing len
    print(f"Total School Records (>{threshold}% Students Missing Any Test): {len(high_schools) if high_schools is not None else 'N/A'}")
    print(f"Total School Records (100% Students Missing Any Test): {len(perfect_schools) if perfect_schools is not None else 'N/A'}")
    print(f"Total Class Records (>{threshold}% Students Missing Any Test):  {len(high_classes) if high_classes is not None else 'N/A'}")
    print(f"Total Class Records (100% Students Missing Any Test): {len(perfect_classes) if perfect_classes is not None else 'N/A'}")
    print("=" * 80)


def print_detailed_tables(high_schools, high_classes, threshold):
    """Prints the detailed tables for high-missing schools and classes."""
    # Prepare display names
    if high_schools is not None:
        high_schools_display = high_schools.rename(columns={
            'schoolId': 'School ID', 'Students_Missing_Tests': 'Students Missing Tests',
            'Total_Students': 'Total Students', 'Missing_Student_%': 'Missing Student %'
        }).sort_values('Missing Student %', ascending=False)
        print(f"\nSchools with >{threshold}% Students Missing At Least One Diagnostic Test")
        print("-" * 80)
        if not high_schools_display.empty:
            print(high_schools_display[['School ID', 'Form', 'Students Missing Tests', 'Total Students', 'Missing Student %']].to_string(index=False))
        else:
            print(f"No schools found meeting the >{threshold}% criteria.")
    else:
        print(f"\nNo school data available to display for >{threshold}% missing.")


    if high_classes is not None:
        high_classes_display = high_classes.rename(columns={
            'schoolId': 'School ID', 'classId': 'Class ID', 'Students_Missing_Tests': 'Students Missing Tests',
            'Total_Students': 'Total Students', 'Missing_Student_%': 'Missing Student %'
        }).sort_values(['School ID', 'Grade', 'Missing Student %'], ascending=[True, True, False])
        print(f"\n\nClasses with >{threshold}% Students Missing At Least One Diagnostic Test")
        print("-" * 80)
        if not high_classes_display.empty:
            print(high_classes_display[['School ID', 'Grade', 'Class ID', 'Form', 'Students Missing Tests', 'Total Students', 'Missing Student %']].to_string(index=False))
        else:
            print(f"No classes found meeting the >{threshold}% criteria.")
    else:
         print(f"\nNo class data available to display for >{threshold}% missing.")


# --- Plotting ---
def create_grade_form_plot(df, grade_order, output_file):
    """Creates and saves the bar chart of missing diagnostics by Grade and Form."""
    if df is None or df.empty or 'Has_Missing_Diagnostics' not in df.columns:
        print("Skipping plot generation due to missing data or prior errors.")
        return

    # Calculate counts for the plot
    missing_by_grade_form = df[df['Has_Missing_Diagnostics']].groupby(['Grade', 'Form']).size().unstack(fill_value=0)
    if 'BOY' not in missing_by_grade_form.columns: missing_by_grade_form['BOY'] = 0
    if 'MOY' not in missing_by_grade_form.columns: missing_by_grade_form['MOY'] = 0

    # Reorder grades
    missing_by_grade_form = missing_by_grade_form.reindex(grade_order).dropna(how='all')

    if missing_by_grade_form.empty:
        print("No data found for the specified grades to generate the plot.")
        return

    # Set up plot
    fig, ax = plt.subplots(figsize=(12, 7))
    grades = missing_by_grade_form.index
    x = np.arange(len(grades))
    width = 0.35

    # --- Corrected Colors (Matching Seaborn Defaults) ---
    colors = {'BOY': '#1f77b4', 'MOY': '#ff7f0e'} # Default Seaborn blue and orange

    # Plot bars
    rects1 = ax.bar(x - width/2, missing_by_grade_form['BOY'], width, label='BOY', color=colors['BOY'])
    rects2 = ax.bar(x + width/2, missing_by_grade_form['MOY'], width, label='MOY', color=colors['MOY'])

    # Customize plot
    ax.set_ylabel('Number of Students with Missing Diagnostics')
    ax.set_title('Students with Missing Diagnostics by Grade and Form')
    ax.set_xticks(x)
    ax.set_xticklabels(grades)
    ax.legend(title="Form")

    # Remove ALL grid lines
    ax.grid(False) # Turn off grid

    # Add value labels on top of bars
    ax.bar_label(rects1, padding=3, fmt='%d') # Use %d for integer format
    ax.bar_label(rects2, padding=3, fmt='%d') # Use %d for integer format

    # Adjust y-axis limits to make space for labels if necessary
    ax.margins(y=0.1) # Add 10% margin to the top of the y-axis

    fig.tight_layout()

    try:
        plt.savefig(output_file, dpi=300)
        print(f"\nVisualization '{output_file}' saved.")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close(fig)


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    df_combined = load_data(BOY_FILE, MOY_FILE)

    # 2. Calculate Missing Indicators
    df_combined = calculate_missing_indicators(df_combined, DIAGNOSTIC_SCORES)

    # 3. Perform Analysis
    school_summary, class_summary = analyze_missing_rates(df_combined)

    # 4. Filter for High Missing Rates
    high_schools, perfect_schools, high_classes, perfect_classes = filter_high_missing(
        school_summary, class_summary, MISSING_THRESHOLD
    )

    # 5. Print Summary Counts (Console Output)
    print_summary_counts(high_schools, perfect_schools, high_classes, perfect_classes, MISSING_THRESHOLD)

    # 6. Print Detailed Tables (Console Output)
    print_detailed_tables(high_schools, high_classes, MISSING_THRESHOLD)

    # 7. Create and Save Plot (File Output)
    create_grade_form_plot(df_combined, GRADE_ORDER, OUTPUT_PLOT_FILE)

    print("\nAnalysis complete.")


