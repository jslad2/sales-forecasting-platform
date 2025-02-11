import streamlit as st
import pandas as pd
import plotly.express as px

# Enable Wide Mode
st.set_page_config(layout="wide", page_title="Sales Insights Dashboard", page_icon="📊")

# --- Custom Styling ---
st.markdown(
    """
    <style>
        .stApp {
            background-color: #F8F9FA;
            color: #333333;
        }
        h1, h2, h3 {
            color: #2B3A42;
            text-align: center;
        }
        .stButton>button {
            background-color: #64D8CB;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #56BBAF;
        }
        .metric-card {
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin: 10px;
        }
        .metric-card h3 {
            color: #2B3A42;
            font-size: 1.5rem;
            margin-bottom: 10px;
        }
        .metric-card p {
            color: #666666;
            font-size: 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown(
    """
    <header style="background-color: #2B3A42; padding: 20px; text-align: center; color: white; border-radius: 12px;">
        <h1 style="margin: 0; font-size: 2.5rem;">Sales Insights Dashboard</h1>
        <p style="margin: 0; font-size: 1.2rem;">Unlock the Power of Your Sales Data</p>
    </header>
    """,
    unsafe_allow_html=True,
)

# --- File Upload ---
st.markdown("### Step 1: Upload Your Sales Data")
uploaded_file = st.file_uploader("Upload your sales data file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load the uploaded file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    
    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    # --- Column Mapping ---
    st.markdown("### Step 2: Map Your Columns")
    
    # Get the column names from the uploaded file
    columns = df.columns.tolist()
    
    # Let users map their columns
    date_column = st.selectbox("Select the Date Column", columns, help="Choose the column that contains date information.")
    sales_column = st.selectbox("Select the Sales Column", columns, help="Choose the column that contains sales data.")
    region_column = st.selectbox("Select the Region Column (optional)", [None] + columns, help="Choose the column that contains region information (if applicable).")
    category_column = st.selectbox("Select the Category Column (optional)", [None] + columns, help="Choose the column that contains product category information (if applicable).")
    
    # Rename columns based on user selection
    df = df.rename(columns={
        date_column: "Date",
        sales_column: "Sales",
        region_column: "Region",
        category_column: "Category",
    })
    
    # Ensure required columns are present
    if "Date" not in df.columns or "Sales" not in df.columns:
        st.error("Please map the required columns: Date and Sales.")
    else:
        st.write("### Mapped Data Preview")
        st.dataframe(df.head())

        # --- Data Processing ---
        # Convert Date column to datetime
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        
        # Drop rows with missing or invalid dates
        df = df.dropna(subset=["Date"])
        
        # Ensure Sales column is numeric
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
        df = df.dropna(subset=["Sales"])
        
        # Handle optional columns
        if "Region" not in df.columns:
            df["Region"] = "All"  # Default value if Region is not provided
        if "Category" not in df.columns:
            df["Category"] = "All"  # Default value if Category is not provided
        
        st.write("### Processed Data Preview")
        st.dataframe(df.head())

        # --- Sidebar Filters ---
        st.sidebar.title("Filters")
        st.sidebar.markdown("Customize the dashboard by selecting filters below.")

        # Date Range Filter
        min_date = df["Date"].min()
        max_date = df["Date"].max()
        date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

        # Region Filter
        if "Region" in df.columns:
            regions = df["Region"].unique().tolist()
            selected_region = st.sidebar.selectbox("Select Region", ["All"] + regions)
        else:
            selected_region = "All"

        # Category Filter
        if "Category" in df.columns:
            categories = df["Category"].unique().tolist()
            selected_category = st.sidebar.selectbox("Select Category", ["All"] + categories)
        else:
            selected_category = "All"

        # Apply Filters
        if selected_region != "All":
            df = df[df["Region"] == selected_region]
        if selected_category != "All":
            df = df[df["Category"] == selected_category]
        df = df[(df["Date"] >= pd.to_datetime(date_range[0])) & (df["Date"] <= pd.to_datetime(date_range[1]))]

        # --- Chart Style Options ---
        st.sidebar.title("Chart Style Options")
        
        # Chart Type
        chart_type = st.sidebar.selectbox("Select Chart Type", ["Line", "Bar", "Area"])

        # Color Scheme
        color_scheme = st.sidebar.selectbox("Select Color Scheme", ["Default", "Dark", "Pastel"])

        # Chart Title and Axis Labels
        chart_title = st.sidebar.text_input("Chart Title", "Daily Sales Trend")
        x_axis_label = st.sidebar.text_input("X-Axis Label", "Date")
        y_axis_label = st.sidebar.text_input("Y-Axis Label", "Sales ($)")

        # --- Generate Dashboard ---
        # --- Key Metrics ---
        st.markdown("## Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                """
                <div class="metric-card">
                    <h3>Total Sales</h3>
                    <p>${:,.0f}</p>
                </div>
                """.format(df["Sales"].sum()),
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="metric-card">
                    <h3>Average Daily Sales</h3>
                    <p>${:,.0f}</p>
                </div>
                """.format(df["Sales"].mean()),
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
                <div class="metric-card">
                    <h3>Max Daily Sales</h3>
                    <p>${:,.0f}</p>
                </div>
                """.format(df["Sales"].max()),
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                """
                <div class="metric-card">
                    <h3>Min Daily Sales</h3>
                    <p>${:,.0f}</p>
                </div>
                """.format(df["Sales"].min()),
                unsafe_allow_html=True,
            )

        # --- Sales Trend Chart ---
        st.markdown(f"## {chart_title}")
        
        # Set color scheme
        if color_scheme == "Dark":
            template = "plotly_dark"
        elif color_scheme == "Pastel":
            template = "plotly_white"
        else:
            template = "plotly"

        # Generate chart based on selected type
        if chart_type == "Line":
            fig = px.line(df, x="Date", y="Sales", title=chart_title, labels={"Sales": y_axis_label}, template=template)
        elif chart_type == "Bar":
            fig = px.bar(df, x="Date", y="Sales", title=chart_title, labels={"Sales": y_axis_label}, template=template)
        elif chart_type == "Area":
            fig = px.area(df, x="Date", y="Sales", title=chart_title, labels={"Sales": y_axis_label}, template=template)

        # Update axis labels
        fig.update_layout(
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Sales by Region ---
        if "Region" in df.columns:
            st.markdown("## Sales by Region")
            region_sales = df.groupby("Region")["Sales"].sum().reset_index()
            fig = px.bar(region_sales, x="Region", y="Sales", title="Total Sales by Region", labels={"Sales": "Sales ($)"}, template=template)
            st.plotly_chart(fig, use_container_width=True)

        # --- Sales by Category ---
        if "Category" in df.columns:
            st.markdown("## Sales by Product Category")
            category_sales = df.groupby("Category")["Sales"].sum().reset_index()
            fig = px.pie(category_sales, values="Sales", names="Category", title="Sales Distribution by Category", template=template)
            st.plotly_chart(fig, use_container_width=True)

        # --- Top Selling Days ---
        st.markdown("## Top Selling Days")
        top_days = df.nlargest(5, "Sales")[["Date", "Sales"]]
        st.dataframe(top_days)

        # --- Download Data ---
        st.markdown("## Download Data")
        st.write("Download the filtered data for further analysis.")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "sales_data.csv", "text/csv")