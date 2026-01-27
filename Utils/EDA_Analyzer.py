import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EDAAnalyzer:
    """
    Comprehensive Exploratory Data Analysis (EDA) toolkit.
    Generates publication-ready visualizations and insights.
    
    Features:
    - Univariate analysis (histograms, boxplots)
    - Bivariate analysis (scatter plots, correlation heatmap)
    - Multivariate analysis (pair plots, PCA visualization)
    - Business-specific insights for pricing data
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
    def generate_complete_eda(self) -> Dict[str, any]:
        """
        Generate complete EDA report with all visualizations.
        
        Returns:
            Dictionary with insights and figure objects
        """
        st.title("🔍 Comprehensive EDA Report")
        
        insights = {}
        
        # 1. Dataset Overview
        insights['overview'] = self.dataset_overview()
        
        # 2. Univariate Analysis
        insights['univariate'] = self.univariate_analysis()
        
        # 3. Bivariate Analysis  
        insights['bivariate'] = self.bivariate_analysis()
        
        # 4. Price Analysis (Business Critical)
        insights['price_analysis'] = self.price_analysis()
        
        # 5. Brand & Platform Analysis
        insights['market_analysis'] = self.market_analysis()
        
        return insights

    def dataset_overview(self) -> Dict:
        """Dataset summary statistics and quality metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(self.df))
        with col2:
            st.metric("Total Features", len(self.df.columns))
        with col3:
            st.metric("Numeric Features", len(self.numeric_cols))
        with col4:
            st.metric("Categorical Features", len(self.categorical_cols))
        
        # Missing values
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            st.warning(f"⚠️ Missing values detected: {missing_data.sum()}")
            st.dataframe(missing_data[missing_data > 0])
        else:
            st.success("✅ No missing values!")
        
        # Data types
        st.subheader("📊 Data Types")
        dtype_summary = self.df.dtypes.value_counts()
        st.bar_chart(dtype_summary)
        
        return {"shape": self.df.shape, "missing": int(missing_data.sum())}

    def univariate_analysis(self) -> Dict:
        """Analyze individual feature distributions."""
        st.header("📈 Univariate Analysis")
        
        # Numeric features
        if self.numeric_cols:
            st.subheader("Numeric Features Distribution")
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for idx, col in enumerate(self.numeric_cols[:6]):
                axes[idx].hist(self.df[col], bins=30, alpha=0.7, edgecolor='black')
                axes[idx].set_title(f'{col} Distribution')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
            
            st.pyplot(fig)
        
        # Categorical features  
        if self.categorical_cols:
            st.subheader("Categorical Features")
            for col in self.categorical_cols[:3]:
                fig = px.bar(
                    x=self.df[col].value_counts().index,
                    y=self.df[col].value_counts().values,
                    title=f'{col} Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        return {"numeric_features": self.numeric_cols, "categorical_features": self.categorical_cols}

    def bivariate_analysis(self) -> Dict:
        """Analyze relationships between features."""
        st.header("🔗 Bivariate Analysis")
        
        # Price vs Rating scatter plot
        if 'price' in self.df.columns and 'rating' in self.df.columns:
            fig = px.scatter(
                self.df, x='rating', y='price', 
                size='review_count', color='platform',
                hover_data=['product_name', 'brand'],
                title="💰 Price vs ⭐ Rating (Size = Reviews)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        if len(self.numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            corr_matrix = self.df[self.numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        return {"correlation_heatmap": corr_matrix.to_dict() if 'corr_matrix' in locals() else {}}

    def price_analysis(self) -> Dict:
        """Business-critical price distribution analysis."""
        st.header("💵 Price Analysis")
        
        if 'price' in self.df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Price distribution
                fig_hist = px.histogram(
                    self.df, x='price',
                    nbins=50,
                    title="Price Distribution",
                    labels={'price': 'Price ($)'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Price statistics
                price_stats = self.df['price'].describe()
                st.metric("Mean Price", f"${price_stats['mean']:.2f}")
                st.metric("Median Price", f"${price_stats['50%']:.2f}")
                st.metric("Price Range", f"${price_stats['min']:.2f} - ${price_stats['max']:.2f}")
            
            # Price by brand
            if 'brand' in self.df.columns:
                st.subheader("Price by Brand")
                brand_price = self.df.groupby('brand')['price'].agg(['mean', 'count']).reset_index()
                brand_price.columns = ['brand', 'avg_price', 'count']
                
                fig_brand = px.bar(
                    brand_price.sort_values('avg_price'), 
                    x='brand', y='avg_price',
                    text='avg_price',
                    title="Average Price by Brand",
                    labels={'avg_price': 'Average Price ($)'}
                )
                st.plotly_chart(fig_brand, use_container_width=True)
        
        return {"price_stats": price_stats.to_dict() if 'price_stats' in locals() else {}}

    def market_analysis(self) -> Dict:
        """Analyze market segments and platforms."""
        st.header("🏪 Market Analysis")
        
        if 'platform' in self.df.columns:
            # Platform comparison
            platform_stats = self.df.groupby('platform').agg({
                'price': ['mean', 'count'],
                'rating': 'mean'
            }).round(2)
            
            platform_stats.columns = ['avg_price', 'product_count', 'avg_rating']
            st.dataframe(platform_stats)
            
            # Platform market share
            fig_pie = px.pie(
                values=platform_stats['product_count'],
                names=platform_stats.index,
                title="Market Share by Platform"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        if 'brand' in self.df.columns and 'platform' in self.df.columns:
            st.subheader("Brand Leadership by Platform")
            crosstab = pd.crosstab(self.df['brand'], self.df['platform'], 
                                 normalize='index').round(3)
            fig_heatmap = px.imshow(
                crosstab,
                title="Brand Distribution Across Platforms",
                aspect="auto",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        return {"platform_stats": platform_stats.to_dict() if 'platform_stats' in locals() else {}}

    def detect_anomalies(self) -> List[Dict]:
        """Detect potential data quality issues."""
        anomalies = []
        
        # Price outliers
        if 'price' in self.df.columns:
            Q1 = self.df['price'].quantile(0.25)
            Q3 = self.df['price'].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df['price'] < Q1 - 1.5*IQR) | 
                              (self.df['price'] > Q3 + 1.5*IQR)]
            
            if len(outliers) > 0:
                anomalies.append({
                    "issue": "PRICE_OUTLIERS",
                    "count": len(outliers),
                    "details": f"Found {len(outliers)} price outliers (IQR method)"
                })
        
        # Low review high rating
        if 'rating' in self.df.columns and 'review_count' in self.df.columns:
            suspicious = self.df[(self.df['rating'] > 4.5) & (self.df['review_count'] < 10)]
            if len(suspicious) > 0:
                anomalies.append({
                    "issue": "SUSPICIOUS_RATINGS",
                    "count": len(suspicious),
                    "details": f"{len(suspicious)} products with high rating (<10 reviews)"
                })
        
        return anomalies

def run_eda_app():
    """Streamlit EDA application."""
    st.set_page_config(
        page_title="🔍 EDA Analyzer",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Exploratory Data Analysis Toolkit")
    
    # File upload
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} records with {len(df.columns)} features")
        
        # Initialize EDA
        analyzer = EDAAnalyzer(df)
        
        # Generate complete analysis
        if st.button("🚀 Generate Complete EDA Report", type="primary"):
            insights = analyzer.generate_complete_eda()
            
            # Anomalies
            anomalies = analyzer.detect_anomalies()
            if anomalies:
                st.error("🚨 Data Quality Issues Detected:")
                for anomaly in anomalies:
                    st.warning(f"⚠️ {anomaly['details']}")
            else:
                st.success("✅ No major data quality issues detected!")
            
            st.session_state.insights = insights
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    **Advanced EDA Toolkit for Data Scientists**
    
    Features:
    • 📊 Complete univariate analysis
    • 🔗 Bivariate relationships & correlations  
    • 💵 Business-focused price analysis
    • 🏪 Market segmentation insights
    • 🚨 Automated anomaly detection
    """)

if __name__ == "__main__":
    run_eda_app()
