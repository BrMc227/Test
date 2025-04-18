import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt # Keep for potential future use if needed
import seaborn as sns # Keep for potential future use if needed

# --- Page Configuration ---
st.set_page_config(page_title="High Missing Test Classes", layout="wide")
st.title('Classes with High Missing Diagnostic Test Rates (>50%)')

# --- Data Loading and Processing ---
@st.cache_data # Cache the data loading and initial processing
def load_and_process_data():
    try:
        boy_df = pd.read_csv('quality_takehome_BOY_file.csv')
        boy_df['Form'] = 'BOY'
        moy_df = pd.read_csv('quality_takehome_MOY_file.csv')
        moy_df['Form'] = 'MOY'
        df = pd.concat([boy_df, moy_df], axis=0, ignore_index=True)
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        return None

    # Define diagnostic scores
    diagnostic_scores = ['Adjusted WCPM Score', 'PA PR', 'SR PR', 'Vocabulary PR', 'Decoding PR']
    # Create shorter, cleaner names for columns
    diagnostic_col_names = {score: f"{score.split(' ')[0]} % Miss" for score in diagnostic_scores}

    # Identify students with any missing diagnostic test
    df['Has_Missing_Diagnostics'] = df[diagnostic_scores].isnull().any(axis=1)

    # --- Class Level Analysis ---
    group_cols = ['schoolId', 'Grade', 'classId', 'Form']

    # 1. Calculate overall missing % (students missing *any* test)
    class_overall_summary = df.groupby(group_cols).agg(
        Total_Students=('classId', 'count'),
        Students_Missing_Any_Test=('Has_Missing_Diagnostics', 'sum')
    ).reset_index()
    class_overall_summary['Overall_Missing_Student_%'] = (class_overall_summary['Students_Missing_Any_Test'] / class_overall_summary['Total_Students'] * 100).round(1)

    # 2. Calculate missing % for *each* specific test
    def calculate_missing_percentage(series):
        if len(series) == 0: return 0.0
        return (series.isnull().sum() / len(series) * 100)

    agg_functions_for_specific_tests = {score: calculate_missing_percentage for score in diagnostic_scores}
    class_specific_summary = df.groupby(group_cols)[diagnostic_scores].agg(agg_functions_for_specific_tests).reset_index()

    # 3. Merge the summaries
    class_combined_summary = pd.merge(
        class_overall_summary,
        class_specific_summary,
        on=group_cols
    )

    # 4. Filter classes where > 50% of students have *at least one* missing test
    high_missing_classes = class_combined_summary[class_combined_summary['Overall_Missing_Student_%'] > 50].copy() # Use .copy() to avoid SettingWithCopyWarning

    # 5. Rename columns for final display
    high_missing_classes = high_missing_classes.rename(columns={
        'schoolId': 'School ID', 'classId': 'Class ID',
        'Students_Missing_Any_Test': 'Students Missing Any Test',
        'Total_Students': 'Total Students', 'Overall_Missing_Student_%': 'Overall Missing Student %'
    })
    high_missing_classes = high_missing_classes.rename(columns=diagnostic_col_names) # Rename specific test columns

    # Define column order for the table
    ordered_cols = ['School ID', 'Grade', 'Class ID', 'Form', 'Total Students', 'Students Missing Any Test', 'Overall Missing Student %'] + list(diagnostic_col_names.values())
    # Ensure all columns exist before ordering
    ordered_cols = [col for col in ordered_cols if col in high_missing_classes.columns]
    high_missing_classes = high_missing_classes[ordered_cols]

    return high_missing_classes

# --- Load Data ---
# This 'processed_data' DataFrame contains ALL classes meeting the initial >50% criteria
processed_data = load_and_process_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Filter by Form (Multiselect) - Affects both table and visualization
form_options = []
if processed_data is not None:
    form_options = sorted(processed_data['Form'].unique())
selected_forms = st.sidebar.multiselect("Select Form(s)", form_options, default=form_options)

# Filter by School (Dropdown with "All") - Primarily affects table
school_options = ['All']
if processed_data is not None:
    school_options = ['All'] + sorted(processed_data['School ID'].unique())
selected_school = st.sidebar.selectbox("Select School (for Table)", school_options, index=0) # Default to "All"

# Filter by Grade (Dropdown with "All") - Primarily affects table
grade_options = ['All']
if processed_data is not None:
    grade_options = ['All'] + sorted(processed_data['Grade'].unique())
selected_grade = st.sidebar.selectbox("Select Grade (for Table)", grade_options, index=0) # Default to "All"

# --- Export Option ---
st.sidebar.header("Export Table Data")
@st.cache_data # Cache the conversion
def convert_df_to_csv(df_to_convert):
   # Ensure the DataFrame isn't empty before converting
   if df_to_convert is None or df_to_convert.empty:
       return "".encode('utf-8') # Return empty bytes if no data
   return df_to_convert.to_csv(index=False).encode('utf-8')

# Placeholder for filtered table data, defined before tabs
filtered_data_for_table = pd.DataFrame()
if processed_data is not None:
    filtered_data_for_table = processed_data.copy() # Initialize before filtering

# --- Main Area with Tabs ---
if processed_data is not None and not processed_data.empty:
    tab1, tab2 = st.tabs(["Detailed Class View", "School Summary"])

    # --- Tab 1: Detailed Class View ---
    with tab1:
        st.subheader("Filtered Class Data (>50% Students Missing Any Test)")

        # Apply filters TO THE TABLE DISPLAY
        filtered_data_for_table = processed_data[processed_data['Form'].isin(selected_forms)].copy() # Start with Form filter

        if selected_school != 'All':
            filtered_data_for_table = filtered_data_for_table[filtered_data_for_table['School ID'] == selected_school]

        if selected_grade != 'All':
            filtered_data_for_table = filtered_data_for_table[filtered_data_for_table['Grade'] == selected_grade]

        # Sort the final filtered data for the table
        filtered_data_for_table = filtered_data_for_table.sort_values(['School ID', 'Grade', 'Overall Missing Student %'], ascending=[True, True, False])

        st.write(f"Displaying {len(filtered_data_for_table)} classes meeting the criteria based on current filters.")

        # Define formatting for percentage columns in the table
        percent_cols_display = ['Overall Missing Student %'] + [col for col in processed_data.columns if '% Miss' in col]
        column_config = {
            col: st.column_config.NumberColumn(format="%.1f%%") for col in percent_cols_display if col in filtered_data_for_table.columns
        }

        # Display the interactive table
        st.dataframe(
            filtered_data_for_table,
            hide_index=True,
            column_config=column_config,
            use_container_width=True
        )

    # --- Tab 2: School Summary Visualization ---
    with tab2:
        st.subheader("Schools with Most High-Missing Classes")
        st.write(f"Count of classes per school where >50% of students missed at least one test (filtered by selected Forms: {', '.join(selected_forms) or 'None'}). The School/Grade filters do not apply here.")

        # Calculate counts based on the initially loaded 'processed_data' filtered only by FORM
        viz_data = processed_data[processed_data['Form'].isin(selected_forms)] # Use form filter for viz

        if not viz_data.empty:
            school_class_counts = viz_data.groupby('School ID').size().reset_index(name='Number of High-Missing Classes')
            school_class_counts = school_class_counts.sort_values('Number of High-Missing Classes', ascending=False)

            # Set School ID as index for bar chart labels
            school_class_counts_chart = school_class_counts.set_index('School ID')

            if not school_class_counts_chart.empty:
                # Use Streamlit's bar chart
                st.bar_chart(school_class_counts_chart['Number of High-Missing Classes'])
            else:
                # This case might occur if filtering by form results in no high-missing classes
                st.info("No high-missing classes found for the selected forms.")
        else:
            st.info("No data available for the selected forms.")

    # --- Enable Export Button ---
    # Placed outside tabs, but uses data filtered for the table
    csv = convert_df_to_csv(filtered_data_for_table)
    st.sidebar.download_button(
       label="📥 Download Filtered Table Data",
       data=csv,
       file_name='high_missing_test_classes_filtered.csv',
       mime='text/csv',
       disabled=filtered_data_for_table.empty # Disable if table is empty
    )


elif processed_data is not None and processed_data.empty:
    st.info("No classes found where >50% of students are missing at least one diagnostic test.")
else:
    # Error message is handled within load_and_process_data
    pass