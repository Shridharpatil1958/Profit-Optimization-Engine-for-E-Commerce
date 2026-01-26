"""
Data cleaning and preprocessing module 
Implements industry-standard data science techniques 
"""

import pandas as pd 
import numpy as np 
from scipy import stats 

class DataProcessor: 
    def __init__(self): 
        self.cleaning_report = [] 
     
    def clean_data(self, df): 
        """
        Complete data cleaning pipeline 
         
        Args: 
            df: Raw DataFrame 
             
        Returns: 
            Cleaned DataFrame and cleaning report 
        """ 
        self.cleaning_report = [] 
        original_shape = df.shape 
         
        # Step 1: Remove duplicates 
        df = self.remove_duplicates(df) 
         
        # Step 2: Handle missing values 
        df = self.handle_missing_values(df) 
         
        # Step 3: Convert and normalize prices 
        df = self.normalize_prices(df) 
         
        # Step 4: Detect and remove outliers 
        df = self.remove_outliers(df) 
         
        # Generate final report 
        final_shape = df.shape 
        self.cleaning_report.append(f"✅ Cleaning complete: {original_shape[0]} → {final_shape[0]} rows")
        return df, self.cleaning_report 
     
    def remove_duplicates(self, df): 
        """
        Remove duplicate listings based on product name and platform 
         
        Business Reasoning: Same product from same platform should appear once 
        """ 
        initial_count = len(df) 
        df = df.drop_duplicates(subset=['product_name', 'platform'], keep='first') 
        removed_count = initial_count - len(df) 
         
        self.cleaning_report.append( 
            f"🔍 Duplicates: Removed {removed_count} duplicate listings " 
            f"(same product + platform)" 
        ) 
         
        return df 
     
    def handle_missing_values(self, df): 
        """
        Handle missing values with appropriate strategies 
         
        Business Reasoning: 
        - Price: Use median (robust to outliers) 
        - Rating: Use platform average 
        - Review count: Use 0 (new products) 
        - Discount: Use 0 (no discount) 
        """ 
        missing_before = df.isnull().sum() 
         
        # Handle price missing values 
        if df['price'].isnull().any(): 
            median_price = df['price'].median() 
            df['price'].fillna(median_price, inplace=True) 
            self.cleaning_report.append( 
                f"💰 Price: Filled {missing_before['price']} missing values with median (${median_price:.2f})" 
            ) 
         
        # Handle rating missing values 
        if 'rating' in df.columns and df['rating'].isnull().any(): 
            platform_avg_rating = df.groupby('platform')['rating'].transform('mean')
            df['rating'].fillna(platform_avg_rating, inplace=True) 
            self.cleaning_report.append( 
                f"⭐ Rating: Filled {missing_before['rating']} missing values with platform average" 
            ) 
         
        # Handle review count missing values 
        if 'review_count' in df.columns and df['review_count'].isnull().any(): 
            df['review_count'].fillna(0, inplace=True) 
            self.cleaning_report.append( 
                f"💬 Reviews: Filled {missing_before['review_count']} missing values with 0 (new products)" 
            ) 
         
        # Handle discount missing values 
        if 'discount' in df.columns and df['discount'].isnull().any(): 
            df['discount'].fillna(0, inplace=True) 
            self.cleaning_report.append( 
                f"🏷️ Discount: Filled {missing_before['discount']} missing values with 0 (no discount)" 
            ) 
         
        return df 
     
    def normalize_prices(self, df): 
        """
        Convert price strings to numeric and normalize 
         
        Business Reasoning: Ensure all prices are in same format for analysis 
        """ 
        # If prices are already numeric, skip 
        if pd.api.types.is_numeric_dtype(df['price']): 
            self.cleaning_report.append("💵 Prices: Already in numeric format") 
            return df 
         
        # Convert string prices to numeric (handle $, commas, etc.) 
        df['price'] = df['price'].astype(str).str.replace('$', '').str.replace(',', '') 
        df['price'] = pd.to_numeric(df['price'], errors='coerce') 
         
        # Remove any rows where price conversion failed 
        invalid_prices = df['price'].isnull().sum() 
        df = df.dropna(subset=['price']) 
         
        self.cleaning_report.append( 
            f"💵 Prices: Converted to numeric format, removed {invalid_prices} invalid entries"
        ) 
         
        return df 
     
    def remove_outliers(self, df, method='both'): 
        """
        Detect and remove outliers using IQR and Z-score methods 
         
        Business Reasoning: Extreme prices may be errors or luxury items 
        that don't represent typical market 
         
        Args: 
            df: DataFrame 
            method: 'iqr', 'zscore', or 'both' 
        """ 
        initial_count = len(df) 
         
        if method in ['iqr', 'both']: 
            df = self._remove_outliers_iqr(df) 
         
        if method in ['zscore', 'both']: 
            df = self._remove_outliers_zscore(df) 
         
        removed_count = initial_count - len(df) 
        self.cleaning_report.append( 
            f"📊 Outliers: Removed {removed_count} outlier prices using {method.upper()} method" 
        ) 
         
        return df 
     
    def _remove_outliers_iqr(self, df): 
        """
        Remove outliers using Interquartile Range (IQR) method 
         
        IQR Method: Remove values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] 
        """ 
        Q1 = df['price'].quantile(0.25) 
        Q3 = df['price'].quantile(0.75) 
        IQR = Q3 - Q1 
         
        lower_bound = Q1 - 1.5 * IQR 
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)] 
         
        return df 
     
    def _remove_outliers_zscore(self, df, threshold=3): 
        """
        Remove outliers using Z-score method 
         
        Z-score Method: Remove values with |z-score| > threshold (typically 3) 
        """ 
        z_scores = np.abs(stats.zscore(df['price'])) 
        df = df[z_scores < threshold] 
         
        return df 
     
    def get_data_quality_report(self, df): 
        """Generate comprehensive data quality report""" 
        report = { 
            "total_records": len(df), 
            "total_features": len(df.columns), 
            "missing_values": df.isnull().sum().to_dict(), 
            "price_range": { 
                "min": df['price'].min(), 
                "max": df['price'].max(), 
                "mean": df['price'].mean(), 
                "median": df['price'].median() 
            }, 
            "platforms": df['platform'].value_counts().to_dict(), 
            "categories": df['category'].value_counts().to_dict() 
        } 
         
        return report[file:8]
