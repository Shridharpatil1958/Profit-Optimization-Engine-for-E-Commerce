"""
Profit Optimization Engine
Finds optimal selling price that maximizes profit
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ProfitOptimizer:
    def __init__(self, ml_model):
        """
        Initialize optimizer with trained ML model
        Args:
            ml_model: Trained MLModels instance
        """
        self.ml_model = ml_model
        self.optimization_results = None

    def optimize_price(self, cost_price, market_data, product_features,
                      price_range_pct=0.3, n_points=100):
        """
        Find optimal price that maximizes profit
        Args:
            cost_price: Product cost price
            market_data: DataFrame with market information
            product_features: Dict with product characteristics
            price_range_pct: Price simulation range (±% from market avg)
            n_points: Number of price points to simulate
        Returns:
            Dictionary with optimization results
        """
        # Get market average price
        market_avg = market_data['price'].mean()
        market_median = market_data['price'].median()

        # Define price range for simulation
        price_min = market_avg * (1 - price_range_pct)
        price_max = market_avg * (1 + price_range_pct)
        
        # Ensure minimum price covers cost
        price_min = max(price_min, cost_price * 1.1)  # At least 10% margin
        
        # Generate price points
        price_points = np.linspace(price_min, price_max, n_points)
        
        # Simulate demand and profit at each price point
        results = []
        for price in price_points:
            # Update features with current price
            features = product_features.copy()
            features['price'] = price
            features['market_avg_price'] = market_avg
            features['market_median_price'] = market_median
            features['price_deviation_pct'] = ((price - market_avg) / market_avg * 100)
            features['value_score'] = (features['rating'] / price * 100)
            
            # Predict demand
            try:
                predicted_demand = self.ml_model.predict_demand(features)
                predicted_demand = max(0, predicted_demand)  # Ensure non-negative
            except:
                # Fallback: simple demand model
                predicted_demand = self._simple_demand_model(price, market_avg, features)
            
            # Calculate profit
            margin = price - cost_price
            profit = margin * predicted_demand
            
            results.append({
                'price': price,
                'predicted_demand': predicted_demand,
                'margin': margin,
                'profit': profit,
                'margin_pct': (margin / price * 100)
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Find optimal price
        optimal_idx = results_df['profit'].idxmax()
        optimal_result = results_df.loc[optimal_idx]
        
        # Classify pricing strategy
        strategy = self._classify_strategy(
            optimal_result['price'],
            optimal_result['margin_pct'],
            market_avg
        )
        
        # Store results
        self.optimization_results = {
            'optimal_price': optimal_result['price'],
            'expected_demand': optimal_result['predicted_demand'],
            'expected_profit': optimal_result['profit'],
            'margin': optimal_result['margin'],
            'margin_pct': optimal_result['margin_pct'],
            'strategy': strategy,
            'market_avg_price': market_avg,
            'market_median_price': market_median,
            'price_deviation': ((optimal_result['price'] - market_avg) / market_avg * 100),
            'simulation_data': results_df,
            'cost_price': cost_price
        }
        
        return self.optimization_results

    def _simple_demand_model(self, price, market_avg, features):
        """
        Simple demand model as fallback
        Demand decreases as price increases relative to market
        """
        base_demand = features.get('rating', 4.0) * np.log1p(features.get('review_count', 100))
        price_factor = (market_avg / price) ** 1.5  # Price elasticity
        demand = base_demand * price_factor
        return max(0, demand)

    def _classify_strategy(self, optimal_price, margin_pct, market_avg):
        """
        Classify pricing strategy based on margin and market position
        Returns:
            Strategy classification string
        """
        price_position = (optimal_price - market_avg) / market_avg * 100
        
        if margin_pct < 20:
            return "Low Margin - High Volume"
        elif margin_pct > 40:
            return "High Margin - Low Volume"
        else:
            return "Balanced"

    def plot_profit_curve(self):
        """Generate profit curve visualization"""
        if self.optimization_results is None:
            raise ValueError("Run optimize_price() first")
        
        df = self.optimization_results['simulation_data']
        optimal_price = self.optimization_results['optimal_price']
        market_avg = self.optimization_results['market_avg_price']
        cost_price = self.optimization_results['cost_price']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Profit vs Price', 'Demand vs Price',
                           'Margin vs Price', 'Profit Components'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "bar"}]]
        )
        
        # 1. Profit curve
        fig.add_trace(
            go.Scatter(x=df['price'], y=df['profit'],
                      mode='lines', name='Profit',
                      line=dict(color='#28a745', width=3)),
            row=1, col=1
        )
        
        # Mark optimal point
        fig.add_trace(
            go.Scatter(x=[optimal_price],
                      y=[self.optimization_results['expected_profit']],
                      mode='markers', name='Optimal Price',
                      marker=dict(size=15, color='red', symbol='star')),
            row=1, col=1
        )
        
        # Mark market average
        fig.add_vline(x=market_avg, line_dash="dash", line_color="blue",
                     annotation_text="Market Avg", row=1, col=1)
        
        # 2. Demand curve
        fig.add_trace(
            go.Scatter(x=df['price'], y=df['predicted_demand'],
                      mode='lines', name='Demand',
                      line=dict(color='#667eea', width=3)),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=[optimal_price],
                      y=[self.optimization_results['expected_demand']],
                      mode='markers', name='Optimal Demand',
                      marker=dict(size=15, color='red', symbol='star')),
            row=1, col=2
        )
        
        # 3. Margin curve
        fig.add_trace(
            go.Scatter(x=df['price'], y=df['margin_pct'],
                      mode='lines', name='Margin %',
                      line=dict(color='#ffc107', width=3)),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=[optimal_price],
                      y=[self.optimization_results['margin_pct']],
                      mode='markers', name='Optimal Margin',
                      marker=dict(size=15, color='red', symbol='star')),
            row=2, col=1
        )
        
        # 4. Profit components bar chart
        components = pd.DataFrame({
            'Component': ['Revenue', 'Cost', 'Profit'],
            'Value': [
                optimal_price * self.optimization_results['expected_demand'],
                cost_price * self.optimization_results['expected_demand'],
                self.optimization_results['expected_profit']
            ]
        })
        
        fig.add_trace(
            go.Bar(x=components['Component'], y=components['Value'],
                  marker_color=['#667eea', '#dc3545', '#28a745'],
                  name='Components'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Profit ($)", row=1, col=1)
        fig.update_xaxes(title_text="Price ($)", row=1, col=2)
        fig.update_yaxes(title_text="Demand (units)", row=1, col=2)
        fig.update_xaxes(title_text="Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="Margin (%)", row=2, col=1)
        
        fig.update_layout(height=800, showlegend=True,
                         title_text="Profit Optimization Analysis")
        
        return fig

    def generate_insights(self):
        """Generate business insights from optimization"""
        if self.optimization_results is None:
            raise ValueError("Run optimize_price() first")
        
        results = self.optimization_results
        insights = []
        
        # Optimal price insight
        insights.append({
            'title': '🎯 Optimal Price',
            'value': f"${results['optimal_price']:.2f}",
            'description': f"This price maximizes profit at ${results['expected_profit']:.2f}"
        })
        
        # Market position
        deviation = results['price_deviation']
        if abs(deviation) < 5:
            position = "market-aligned"
        elif deviation > 0:
            position = f"{deviation:.1f}% above market (premium)"
        else:
            position = f"{abs(deviation):.1f}% below market (competitive)"
        
        insights.append({
            'title': '📊 Market Position',
            'value': position,
            'description': f"Market average is ${results['market_avg_price']:.2f}"
        })
        
        # Margin insight
        insights.append({
            'title': '💰 Profit Margin',
            'value': f"{results['margin_pct']:.1f}%",
            'description': f"Margin of ${results['margin']:.2f} per unit"
        })
        
        # Expected demand
        insights.append({
            'title': '📈 Expected Demand',
            'value': f"{results['expected_demand']:.0f} units",
            'description': "Predicted based on ML model"
        })
        
        # Strategy
        strategy_emoji = {
            "Low Margin - High Volume": "🏪",
            "Balanced": "⚖️",
            "High Margin - Low Volume": "💎"
        }
        insights.append({
            'title': f"{strategy_emoji.get(results['strategy'], '📋')} Pricing Strategy",
            'value': results['strategy'],
            'description': self._get_strategy_description(results['strategy'])
        })
        
        return insights

    def _get_strategy_description(self, strategy):
        """Get description for pricing strategy"""
        descriptions = {
            "Low Margin - High Volume": "Focus on volume sales with competitive pricing. Good for market penetration.",
            "Balanced": "Balanced approach between margin and volume. Suitable for stable market position.",
            "High Margin - Low Volume": "Premium positioning with higher margins. Targets quality-conscious customers."
        }
        return descriptions.get(strategy, "Custom pricing strategy")

    def get_recommendations(self):
        """Generate actionable recommendations"""
        if self.optimization_results is None:
            raise ValueError("Run optimize_price() first")
        
        results = self.optimization_results
        recommendations = []
        
        # Price recommendation
        recommendations.append(
            f"✅ **Set price at ${results['optimal_price']:.2f}** to maximize profit"
        )
        
        # Strategy recommendation
        if results['strategy'] == "Low Margin - High Volume":
            recommendations.append(
                "📢 **Marketing Focus**: Emphasize value for money and competitive pricing"
            )
            recommendations.append(
                "📦 **Operations**: Prepare for higher volume, optimize supply chain"
            )
        elif results['strategy'] == "High Margin - Low Volume":
            recommendations.append(
                "✨ **Marketing Focus**: Highlight premium quality and unique features"
            )
            recommendations.append(
                "🎯 **Target Audience**: Focus on quality-conscious, less price-sensitive customers"
            )
        else:
            recommendations.append(
                "🎯 **Marketing Focus**: Balance between value and quality messaging"
            )
            recommendations.append(
                "📊 **Monitor**: Track both volume and margin metrics closely"
            )
        
        # Competitive positioning
        if results['price_deviation'] > 10:
            recommendations.append(
                "⚠️ **Watch Competition**: Your price is significantly above market average"
            )
        elif results['price_deviation'] < -10:
            recommendations.append(
                "💡 **Consider**: You may have room to increase price without losing demand"
            )
        
        return recommendations
