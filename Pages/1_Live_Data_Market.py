"""
Live Market Data Page
Fetch and display real-time competitor pricing data
"""

import streamlit as st
import pandas as pd
from utils.api_handler import APIHandler
from utils.data_processor import DataProcessor
from utils.styles import get_custom_css, create_header, create_metric_card, create_info_box
import config

# Page configuration
st.set_page_config(
    page_title="Live Market Data",
    page_icon="📊",
    layout="wide"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown(
    create_header(
        "📊 Live Market Data Collection",
        "Fetch real-time competitor pricing from e-commerce platforms"
    ),
    unsafe_allow_html=True
)

# Initialize handlers
if 'api_handler' not in st.session_state:
    st.session_state.api_handler = APIHandler()

if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()

# Input section
st.markdown("### 🔍 Search Parameters")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    categories = st.session_state.api_handler.get_categories()
    selected_category = st.selectbox(
        "Product Category",
        categories,
        help="Select the product category to search"
    )

with col2:
    search_query = st.text_input(
        "Search Term",
        value="laptop",
        help="Enter product name or keyword"
    )

with col3:
    use_cache = st.checkbox(
        "Use Cache",
        value=True,
        help="Use cached data if available (faster)"
    )

# Fetch button
if st.button("🔄 Fetch Market Data", type="primary", use_container_width=True):
    with st.spinner("Fetching market data..."):
        try:
            # Fetch data
            raw_data = st.session_state.api_handler.fetch_market_data(
                query=search_query,
                category=selected_category.lower(),
                use_cache=use_cache
            )
            
            st.session_state.raw_market_data = raw_data
            st.success(f"✅ Fetched {len(raw_data)} products from market")
            
            # Clean data
            with st.spinner("Cleaning and processing data..."):
                cleaned_data, cleaning_report = st.session_state.data_processor.clean_data(raw_data)
                st.session_state.cleaned_market_data = cleaned_data
                st.session_state.cleaning_report = cleaning_report

        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")

# Display data if available
if 'raw_market_data' in st.session_state:
    st.markdown("---")
    st.markdown("### 📋 Raw Market Data")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    raw_data = st.session_state.raw_market_data
    
    with col1:
        st.markdown(
            create_metric_card(
                "Total Products",
                f"{len(raw_data)}",
                "Products fetched from market"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            create_metric_card(
                "Avg Price",
                f"${raw_data['price'].mean():.2f}",
                f"Range: ${raw_data['price'].min():.2f} - ${raw_data['price'].max():.2f}"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            create_metric_card(
                "Avg Rating",
                f"{raw_data['rating'].mean():.1f} ⭐",
                f"{(raw_data['rating'] >= 4.0).sum()} products with 4+ stars"
            ),
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            create_metric_card(
                "Platforms",
                f"{raw_data['platform'].nunique()}",
                f"{', '.join(raw_data['platform'].unique()[:3])}"
            ),
            unsafe_allow_html=True
        )
    
    # Display table
    st.dataframe(
        raw_data.style.format({
            'price': '${:.2f}',
            'discount': '{:.1f}%',
            'rating': '{:.1f}'
        }),
        use_container_width=True,
        height=400
    )

# Display cleaned data if available
if 'cleaned_market_data' in st.session_state:
    st.markdown("---")
    st.markdown("### 🧹 Cleaned & Processed Data")
    
    # Cleaning report
    st.markdown("#### Data Cleaning Report")
    
    for report_item in st.session_state.cleaning_report:
        st.markdown(f"- {report_item}")
    
    # Display cleaned data
    cleaned_data = st.session_state.cleaned_market_data
    
    st.markdown("#### Cleaned Dataset")
    st.dataframe(
        cleaned_data.style.format({
            'price': '${:.2f}',
            'discount': '{:.1f}%',
            'rating': '{:.1f}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Data quality report
    with st.expander("📊 Data Quality Report"):
        quality_report = st.session_state.data_processor.get_data_quality_report(cleaned_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Overview**")
            st.write(f"Total Records: {quality_report['total_records']}")
            st.write(f"Total Features: {quality_report['total_features']}")
            
            st.markdown("**Price Statistics**")
            st.write(f"Min: ${quality_report['price_range']['min']:.2f}")
            st.write(f"Max: ${quality_report['price_range']['max']:.2f}")
            st.write(f"Mean: ${quality_report['price_range']['mean']:.2f}")
            st.write(f"Median: ${quality_report['price_range']['median']:.2f}")
        
        with col2:
            st.markdown("**Platform Distribution**")
            platform_df = pd.DataFrame(
                list(quality_report['platforms'].items()),
                columns=['Platform', 'Count']
            )
            st.dataframe(platform_df, use_container_width=True)
    
    # Download button
    csv = cleaned_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Cleaned Data (CSV)",
        data=csv,
        file_name=f"market_data_{search_query}_{selected_category}.csv",
        mime="text/csv"
    )

# Help section
with st.expander("ℹ️ Help & Information"):
    st.markdown("""
    ### How to Use This Page
    
    1. **Select Category**: Choose the product category you want to analyze
    2. **Enter Search Term**: Type the product name or keyword
    3. **Fetch Data**: Click the button to retrieve market data
    4. **Review Results**: Examine raw and cleaned data
    
    ### Data Sources
    
    - **API Integration**: Connects to e-commerce APIs (RapidAPI, Amazon, eBay, etc.)
    - **Rate Limiting**: Respects API rate limits to avoid blocking
    - **Caching**: Stores recent results to reduce API calls
    - **Fallback**: Uses sample data if API is unavailable
    
    ### Data Cleaning Process
    
    1. **Duplicate Removal**: Removes same product from same platform
    2. **Missing Values**: Fills gaps using median (price) or platform average (rating)
    3. **Outlier Detection**: Uses IQR and Z-score methods to remove extreme values
    4. **Normalization**: Converts all prices to numeric format
    
    ### Business Reasoning
    
    - **Why remove duplicates?** Same product shouldn't be counted twice
    - **Why use median for prices?** More robust to extreme values than mean
    - **Why remove outliers?** Luxury/error prices don't represent typical market
    - **Why normalize?** Ensures consistent analysis across platforms
    """)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Current Session")
    
    if 'cleaned_market_data' in st.session_state:
        data = st.session_state.cleaned_market_data
        st.metric("Products Loaded", len(data))
        st.metric("Avg Price", f"${data['price'].mean():.2f}")
        st.metric("Price Range", f"${data['price'].min():.2f} - ${data['price'].max():.2f}")
    else:
        st.info("No data loaded yet. Fetch market data to begin.")
    
    st.markdown("---")
    st.markdown("### 🎯 Next Steps")
    st.markdown("""
    After fetching market data:
    1. Go to **Profit Simulator**
    2. Input your cost price
    3. Run optimization
    """)
