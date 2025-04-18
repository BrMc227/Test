## Script 1: Amira_EDA.py

# Amira BOY/MOY Diagnostic Data Quality EDA

## 📍 Overview

This Python script (`amira_EDA.py`) performs an exploratory data analysis (EDA) of Amira Benchmark Assessment BOY (Beginning of Year) and MOY (Middle of Year) data to:

- Assess missing component test data (Adjusted WCPM Score, PA PR, SR PR, Vocabulary PR, Decoding PR)
- Evaluate missing data patterns by school, class, and grade
- Identify classes and schools with high rates of missing diagnostics
- Visualize diagnostic completion trends
- Generate an automated Sweetviz comparison report

---

## 🛠️ Requirements

- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sweetviz`
- Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn sweetviz
```

---

## 📂 Files Needed

- `quality_takehome_BOY_file.csv`
- `quality_takehome_MOY_file.csv`

*(Place in the same directory as the script.)*

---

## 🚀 How to Run

```bash
python amira_EDA.py
```

Script outputs:

- Console summaries of missing diagnostic rates by class and school
- Sweetviz HTML report: `Amira_BOY_MOY_Comparison.html`
- Bar charts:
  - `missing_components_by_form.png`
  - `missing_diagnostics_by_grade_form.png`
- Detailed printed reports for high-missing classes and schools

---

## 📊 Key Functions

- **load\_data**: Loads and combines BOY and MOY datasets.
- **calculate\_missing\_indicators**: Flags students with any missing diagnostic.
- **analyze\_missing\_rates**: Summarizes missingness at the school and class levels.
- **filter\_high\_missing**: Identifies schools/classes with >50% missing rates.
- **create\_grade\_form\_plot**: Creates a missing diagnostics visualization by grade.

---

## 🧹 Notes

- Missing rates are flagged at >50% and 100% thresholds.
- The diagnostic score columns must exist in both BOY and MOY files.
- Visualizations and reports are auto-saved to project directory.

---

## 🔮 Future Enhancements

- Add CLI options for dynamic threshold setting
- Build interactive dashboards for missing diagnostics
- Perform longitudinal tracking across multiple testing windows

---



##  Script 2:  score_trends_analysis.py

## 📍 Overview
This Python script analyzes Amira BOY (Beginning of Year) and MOY (Middle of Year) Benchmark Assessment data to:
- Identify missing diagnostic component test scores
- Categorize classes by data completeness
- Compare ARM scores across groups
- Visualize ARM trends
- Conduct t-tests on ARM performance by completeness

## 🛠️ Requirements
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `scipy`
- Install dependencies:

```bash
pip install pandas numpy matplotlib scipy
```

## 📂 Files Needed
- `quality_takehome_BOY_file.csv`
- `quality_takehome_MOY_file.csv`

*(Place in the same directory as the script.)*

## 🚀 How to Run

```bash
python your_script_name.py
```

Script outputs:
- Cleaned and categorized dataset
- ARM score trend comparisons
- T-test results comparing Complete, Partial, and High Missing classes
- Visualizations saved to `/arm_comparison_plots/`

## 📊 Outputs
- Grouped bar charts:
  - `comparison_avg_boy_arm.png`
  - `comparison_avg_moy_arm.png`
  - `comparison_avg_arm_change.png`
- Printed t-test results
- Summary tables by grade and completeness

## 🧹 Notes
- Missing diagnostic data is auto-coerced to `NaN`.
- The script expects full BOY and MOY datasets.
- ARM score column: `ARM`

## 🔮 Future Enhancements
- Dynamic file input (via command-line arguments)
