"""
Wiki / Knowledge Base Page
Provides detailed documentation for the Report Monitor
 application.
"""
import streamlit as st

st.set_page_config(page_title="Wiki & Documentation", layout="wide")

# Inject Custom CSS
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.title("Wiki & Documentation")
st.markdown("Welcome to the Report Monitor knowledge base. Below is the complete documentation.")

# Create tabs for organized viewing
tabs = st.tabs([
    "Overview", 
    "KPIs", 
    "Filters", 
    "Visualizations", 
    "Anomaly & Drift", 
    "Forecasting", 
    "Technical & Data",
    "Glossary & Support"
])

with tabs[0]:
    st.markdown("""
    ## Overview
    
    The **Report Monitor** is a Streamlit dashboard designed to monitor and optimize report execution performance in an Infrastructure-as-a-Service (IaaS) environment. It provides real-time insights into report execution times, failure rates, and performance anomalies.

    ### What Problems Does It Solve?

    - **Performance Monitoring**: Track which reports are running slowly
    - **Failure Detection**: Identify reports that frequently fail
    - **Anomaly Detection**: Automatically detect unusual execution times
    - **Trend Analysis**: Understand execution patterns over time
    - **Resource Optimization**: Identify peak usage hours for better resource allocation
    """)

with tabs[1]:
    st.markdown("""
    ## KPIs (Key Performance Indicators)

    The dashboard displays 4 main KPIs at the top of the page:

    ### 1. Total Executions
    **What it shows**: The total number of report executions in the selected time period.
    
    **Why it matters**: Helps you understand the overall workload and activity level.
    
    **Example**: `15,234` means 15,234 reports were executed in the filtered period.

    ---

    ### 2. Average Duration (Success Only)
    **What it shows**: The average execution time (in seconds) for **successfully completed** reports only.
    
    **Formula**: 
    ```
    Average Duration = Sum of all successful execution times / Number of successful executions
    ```
    
    **Why it matters**: 
    - Indicates overall system performance
    - Failed reports are excluded to avoid skewing the metric
    - Lower is better (faster execution)
    
    **Example**: `45.23s` means successful reports take an average of 45.23 seconds to complete.

    ---

    ### 3. Failure Rate
    **What it shows**: The percentage of reports that failed out of all executions.
    
    **Formula**:
    ```
    Failure Rate = (Number of failed reports / Total executions) × 100
    ```
    
    **Why it matters**: 
    - High failure rates indicate system problems
    - Helps prioritize which reports need attention
    - Lower is better
    
    **Example**: `3.2%` means 3.2% of all report executions failed.
    
    **Color coding**: The delta is shown as "inverse" - meaning increases are bad (red) and decreases are good (green).

    ---

    ### 4. Variability (MAD median)
    **What it shows**: A robust measure of how consistent report execution times are.
    
    **What is MAD?**: 
    - **MAD** = Median Absolute Deviation
    - A statistical measure that's resistant to outliers
    - Measures how spread out the data is from the median
    
    **How it's calculated**:
    1. For each report, calculate the median execution time
    2. For each execution, calculate: `|execution_time - median|`
    3. Take the median of those absolute deviations (this is the MAD for that report)
    4. Take the median of all report MADs (this is the global variability)
    
    **Formula**:
    ```
    For each report:
      MAD = median(|duration - median(duration)|)
    
    Global Variability = median(all MADs)
    ```
    
    **Why it matters**:
    - Low variability = predictable, consistent performance
    - High variability = unpredictable execution times (could indicate resource contention)
    - More robust than standard deviation (not affected by extreme outliers)
    
    **Example**: `12.45s` means the typical deviation from median execution time is 12.45 seconds.
    """)

with tabs[2]:
    st.markdown("""
    ## Filters

    Located in the sidebar, filters allow you to focus on specific data:

    ### Date Range
    - Select start and end dates
    - Default: All available data
    - Use to analyze specific time periods

    ### Report Types
    - Filter by report category/type
    - Multiple selection allowed
    - Default: All types selected

    ### Status
    - **success**: Completed successfully
    - **failed**: Execution failed
    - **cancelled**: Manually cancelled
    - Default: All statuses selected

    ### Reports (Top 200 by volume)
    - Shows only the 200 most frequently executed reports
    - Optional filter to focus on specific reports
    - Default: None selected (shows all)
    """)

with tabs[3]:
    st.markdown("""
    ## Charts and Visualizations

    ### 1. Execution Trends
    **What it shows**: Line chart showing daily execution counts by status (success/failed/cancelled).
    
    **How to use it**:
    - Identify trends over time
    - Spot sudden spikes in failures
    - Understand workload patterns

    ---

    ### 2. Top 10 Slowest Reports
    **What it shows**: Bar chart of the 10 reports with the highest average execution time.
    
    **How to use it**:
    - Prioritize optimization efforts
    - Identify reports that need performance tuning
    - Compare relative performance

    ---

    ### 3. Top 10 Reports with Most Failures
    **What it shows**: Bar chart of reports with the highest failure counts.
    
    **How to use it**:
    - Identify problematic reports
    - Prioritize bug fixes
    - Understand reliability issues

    ---

    ### 4. Duration Distribution
    **What it shows**: Histogram showing how execution times are distributed.
    
    **Interactive**: Adjust bin size (1-120 seconds) to change granularity.
    
    **How to use it**:
    - Understand typical execution time ranges
    - Identify if there are multiple performance "modes"
    - Spot outliers

    ---

    ### 5. Hourly Saturation
    **What it shows**: Area chart showing execution counts by hour of day (0-23).
    
    **How to use it**:
    - Identify peak usage hours
    - Plan maintenance windows during low-activity periods
    - Optimize resource allocation

    ---

    ### 6. Hour vs Weekday Heatmap
    **What it shows**: Table showing execution counts by weekday (0=Monday, 6=Sunday) and hour.
    
    **How to use it**:
    - Identify weekly patterns
    - Understand when reports are most frequently run
    - Plan capacity based on usage patterns
    """)

with tabs[4]:
    st.markdown("""
    ## Anomaly Detection

    Anomaly detection automatically identifies report executions that took unusually long compared to their typical behavior.

    ### Two Detection Methods

    #### 1. Robust by ReportName (MAD)
    **How it works**:
    1. For each report, calculate the median execution time
    2. Calculate the MAD (Median Absolute Deviation)
    3. For each execution, calculate a "robust Z-score":
       ```
       z_robust = 0.6745 × (execution_time - median) / MAD
       ```
    4. Flag executions where `z_robust > threshold`
    
    **When to use**: 
    - General anomaly detection
    - When you want to compare each report to its own historical performance
    
    **Example**: If a report normally takes 30 seconds (median) with MAD of 5 seconds, an execution of 60 seconds would have:
    ```
    z_robust = 0.6745 × (60 - 30) / 5 = 4.05
    ```
    This would be flagged as an anomaly (if threshold is 3.5).

    ---

    #### 2. Contextual (ReportName + Hour)
    **How it works**:
    - Same as method 1, but calculates median and MAD **per report AND per hour**
    - Accounts for time-of-day performance variations
    
    **When to use**:
    - When reports perform differently at different times of day
    - When system load varies significantly by hour
    - More precise but requires more historical data
    
    **Example**: A report might be fast at 2 AM (low load) but slower at 2 PM (high load). Contextual detection accounts for this.

    ---

    ### Sensitivity Threshold
    **Range**: 2.0 to 10.0 (default: 3.5)
    
    **What it means**:
    - **Lower values** (2.0-3.0): More sensitive, catches more anomalies (including minor ones)
    - **Higher values** (7.0-10.0): Less sensitive, only catches extreme anomalies
    
    **Recommended settings**:
    - **3.5**: Good balance for most use cases
    - **2.5**: If you want early warnings
    - **5.0**: If you only care about severe anomalies

    ---

    ## Drift Analysis

    Drift detection identifies when report performance is **changing over time** (getting slower or faster).

    ### How It Works
    
    **Concept**: Compare recent performance to historical performance.
    
    **Method**:
    1. **Window A** (Recent): Last 7 days
    2. **Window B** (Historical): 30 days before Window A
    3. Compare quantiles (p50 = median, p90 = 90th percentile) between windows
    
    **Metrics shown**:
    - **A_p50**: Median duration in recent window
    - **B_p50**: Median duration in historical window
    - **delta_p50**: Change in median (`A_p50 - B_p50`)
    - **A_p90**: 90th percentile in recent window
    - **B_p90**: 90th percentile in historical window
    - **delta_p90**: Change in 90th percentile
    - **ratio_p90**: Ratio of recent to historical 90th percentile (`A_p90 / B_p90`)
    
    ### Interpreting Results
    
    **Positive drift** (getting slower):
    - `delta_p50 > 0`: Median execution time increased
    - `ratio_p90 > 1.0`: 90th percentile is slower than before
    - **Action**: Investigate performance degradation
    
    **Negative drift** (getting faster):
    - `delta_p50 < 0`: Median execution time decreased
    - `ratio_p90 < 1.0`: 90th percentile is faster than before
    - **Action**: Good! Performance improved (or verify data quality)
    """)

with tabs[5]:
    st.markdown("""
    ## Advanced Forecasting (Day 3)

    The dashboard includes a dedicated forecasting module using **Meta's Prophet**, enhanced with hybrid machine learning techniques.

    ### 1. The Prophet Model
    Prophet is an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
    
    **Key Hyperparameters (Advanced Mode):**
    *   **Changepoint Prior Scale**: Controls trend flexibility. 
        *   *High (0.15+)*: Model reacts quickly to shifts but may overfit noise.
        *   *Low (0.001-0.01)*: Model stays on a rigid path.
    *   **Seasonality Mode**:
        *   *Additive*: Seasonal effects are constant.
        *   *Multiplicative*: Seasonal spikes grow as the total volume grows (better for exponential growth).
    *   **Log Transform (log1p)**: Highly recommended for report count data. It stabilizes variance and handles large spikes by calculating errors in percentage terms.

    ---

    ### 2. Hybrid Mode (Lag-1 Regressor)
    This is the "secret weapon" to beat the baseline. 
    *   **The Problem**: In highly autocorrelated data, the value at hour *t* is almost always similar to hour *t-1*. A standard Prophet model only looks at the "logical clock" (time/date), which is why naive baselines often win.
    *   **The Solution**: We add the previous hour's value (`y` at `t-1`) as a continuous regressor to the Prophet model. This combines the "memory" of the baseline with the "logic" (seasonality/holidays) of Prophet.

    ---

    ### 3. Expert Preset Configuration
    To simplify usage, an **Expert Preset** is available that automatically applies:
    - **Daily, Weekly, and Yearly Seasonality**: On.
    - **Log Transformation**: On.
    - **Hybrid Lag-1 Regressor**: On.
    - **Flexible Trend (0.15)**: To capture sudden system shifts.
    - **US Holidays**: Included automatically.

    ---

    ### 4. How to Beat the Baseline
    To outperform the "last hour" naive model:
    1.  **Check Autocorrelation**: Use the Advanced Diagnostics to see if Lag 1 is high (>0.8).
    2.  **Enable Hybrid Mode**: Essential if autocorrelation is high.
    3.  **Apply Log Transform**: Crucial if you have spikes that reach 10x the median value.
    4.  **Use Multiplicative Seasonality**: If spikes are bigger on high-volume days.
    """)

with tabs[6]:
    st.markdown("""
    ## Data Quality Checks

    Located in the expandable "Calidad de datos / Health checks" section.

    ### What It Shows
    
    1. **Current Window**:
       - First and last timestamp in filtered data
       - Helps verify you're looking at the right time period
    
    2. **Null Counts Table**:
       - Shows number of null/missing values per column
       - Helps identify data quality issues
    
    ### Data Sanitization
    
    The app automatically filters out:
    - Records with null `duration_seconds`
    - Records with negative or zero duration
    - Records with duration > 6 hours (configurable in `config.py`)
    
    **Why**: These are likely data errors or system issues, not real executions.

    ---

    ## Technical Details

    ### Architecture
    
    ```
    app.py                    # Main orchestration
    ├── config.py            # Configuration & settings
    ├── db/
    │   └── database.py      # Database queries
    ├── utils/
    │   ├── data_processing.py   # Data transformations
    │   └── statistics.py        # Statistical functions
    └── components/
    │   ├── filters.py           # Sidebar filters
    │   ├── kpis.py             # KPI calculations
    │   ├── charts.py           # Visualizations
    │   └── anomaly_detection.py # Anomaly & drift detection
    ```
    
    ### Data Flow
    
    1. **Load**: Query database for report execution data
    2. **Transform**: Add time features (date, hour, weekday)
    3. **Sanitize**: Remove invalid records
    4. **Filter**: Apply user-selected filters
    5. **Analyze**: Calculate KPIs, detect anomalies, analyze drift
    6. **Visualize**: Render charts and tables
    
    ### Performance Optimizations
    
    - **Caching**: Database queries cached for 10 minutes (configurable)
    - **Lazy Loading**: Charts only render when visible
    - **Efficient Queries**: Uses Polars for fast data processing
    """)

with tabs[7]:
    st.markdown("""
    ## Glossary
    *   **MAE**: Mean Absolute Error (average magnitude of errors).
    *   **RMSE**: Root Mean Square Error (penalizes large errors more heavily).
    *   **Autocorrelation**: How much the current value depends on past values.
    *   **Lag**: The time delay (e.g., Lag 1 = 1 hour ago).
    *   **Residual**: The difference between Actual and Predicted (`Actual - Pred`).

    ---

    ## Support
    For issues or questions regarding the forecasting logic:
    1. Review the **Forecasting Wiki** expander within the app.
    2. Check the `utils/prophet_forecast.py` module for implementation details.
    
    **Last Updated**: February 2026
    """)
