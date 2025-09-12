import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import shap
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for file output
import matplotlib
matplotlib.use('Agg')

def load_and_prepare_data():
    """Load and prepare data for analysis"""
    print("Loading data...")
    static_df = pd.read_csv('synthetic_static_data.csv')
    ts_data = np.load('synthetic_ts_data.npy')
    
    # Feature names
    static_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'glucose', 'smoking']
    ts_features = ['HR', 'Sys_BP', 'Dia_BP', 'SpO2', 'RR_interval', 'QRS_amplitude', 'Acc_X', 'Acc_Y', 'Acc_Z']
    
    return static_df, ts_data, static_features, ts_features

def analyze_static_features(static_df, static_features):
    """Analyze static features for stroke prediction"""
    print("Analyzing static features...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Static Features Analysis for Stroke Classification', fontsize=16, fontweight='bold')
    
    stroke_patients = static_df[static_df['stroke'] == 1]
    normal_patients = static_df[static_df['stroke'] == 0]
    
    for i, feature in enumerate(static_features):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Box plot
        data_to_plot = [normal_patients[feature], stroke_patients[feature]]
        box_plot = ax.boxplot(data_to_plot, labels=['Normal', 'Stroke'], patch_artist=True)
        
        # Color the boxes
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_title(f'{feature.replace("_", " ").title()}', fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add statistical test
        stat, p_value = stats.ttest_ind(normal_patients[feature], stroke_patients[feature])
        significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        ax.text(0.02, 0.98, f'p-value: {significance}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('static_features_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = static_df[static_features + ['stroke']].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix with Stroke', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return stroke_patients, normal_patients

def analyze_time_series_features(ts_data, static_df, ts_features):
    """Analyze time series features"""
    print("Analyzing time series features...")
    
    # Calculate statistical features for each patient
    stroke_indices = static_df['stroke'] == 1
    normal_indices = static_df['stroke'] == 0
    
    stroke_ts = ts_data[stroke_indices]
    normal_ts = ts_data[normal_indices]
    
    # Calculate mean, std, min, max for each feature
    stats_features = []
    feature_names = []
    
    for stat_name in ['mean', 'std', 'min', 'max']:
        for i, feature in enumerate(ts_features):
            if stat_name == 'mean':
                stroke_vals = np.mean(stroke_ts[:, :, i], axis=1)
                normal_vals = np.mean(normal_ts[:, :, i], axis=1)
            elif stat_name == 'std':
                stroke_vals = np.std(stroke_ts[:, :, i], axis=1)
                normal_vals = np.std(normal_ts[:, :, i], axis=1)
            elif stat_name == 'min':
                stroke_vals = np.min(stroke_ts[:, :, i], axis=1)
                normal_vals = np.min(normal_ts[:, :, i], axis=1)
            else:  # max
                stroke_vals = np.max(stroke_ts[:, :, i], axis=1)
                normal_vals = np.max(normal_ts[:, :, i], axis=1)
            
            stats_features.append((normal_vals, stroke_vals))
            feature_names.append(f'{feature}_{stat_name}')
    
    # Create visualization
    n_features = len(ts_features)
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Time Series Features Analysis (Mean Values)', fontsize=16, fontweight='bold')
    
    for i in range(n_features):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Get mean values for this feature
        normal_means = np.mean(normal_ts[:, :, i], axis=1)
        stroke_means = np.mean(stroke_ts[:, :, i], axis=1)
        
        # Create violin plot
        data_to_plot = [normal_means, stroke_means]
        parts = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True, showmedians=True)
        
        # Color the violins
        parts['bodies'][0].set_facecolor('lightblue')
        parts['bodies'][1].set_facecolor('lightcoral')
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Normal', 'Stroke'])
        ax.set_title(f'{ts_features[i]} (Mean)', fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Statistical test
        stat, p_value = stats.ttest_ind(normal_means, stroke_means)
        significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        ax.text(0.02, 0.98, f'p-value: {significance}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('timeseries_features_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats_features, feature_names

def create_feature_importance_analysis(static_df, ts_data, static_features, ts_features):
    """Create comprehensive feature importance analysis"""
    print("Creating feature importance analysis...")
    
    # Prepare data
    X_static = static_df[static_features].values
    y = static_df['stroke'].values
    
    # Extract time series statistical features
    n_samples, n_timesteps, n_ts_features = ts_data.shape
    ts_stats = []
    ts_feature_names = []
    
    for stat_name in ['mean', 'std', 'min', 'max']:
        for i, feature in enumerate(ts_features):
            if stat_name == 'mean':
                values = np.mean(ts_data[:, :, i], axis=1)
            elif stat_name == 'std':
                values = np.std(ts_data[:, :, i], axis=1)
            elif stat_name == 'min':
                values = np.min(ts_data[:, :, i], axis=1)
            else:  # max
                values = np.max(ts_data[:, :, i], axis=1)
            
            ts_stats.append(values)
            ts_feature_names.append(f'{feature}_{stat_name}')
    
    X_ts_stats = np.array(ts_stats).T
    
    # Combine all features
    X_combined = np.hstack([X_static, X_ts_stats])
    all_feature_names = static_features + ts_feature_names
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_combined, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    # Plot feature importance
    plt.figure(figsize=(12, 16))
    colors = ['red' if name in static_features else 'blue' for name in feature_importance_df['feature']]
    bars = plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'], color=colors, alpha=0.7)
    
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
    plt.xlabel('Feature Importance', fontweight='bold')
    plt.title('Feature Importance for Stroke Classification\n(Red: Static Features, Blue: Time Series Features)', 
              fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create top features analysis
    top_features = feature_importance_df.tail(10)
    plt.figure(figsize=(10, 6))
    colors = ['red' if name in static_features else 'blue' for name in top_features['feature']]
    plt.bar(range(len(top_features)), top_features['importance'], color=colors, alpha=0.7)
    plt.xticks(range(len(top_features)), top_features['feature'], rotation=45, ha='right')
    plt.ylabel('Feature Importance', fontweight='bold')
    plt.title('Top 10 Most Important Features for Stroke Classification', fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('top_features_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return feature_importance_df

def create_age_analysis(static_df):
    """Create detailed age analysis"""
    print("Creating age analysis...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Age distribution
    axes[0].hist(static_df[static_df['stroke'] == 0]['age'], bins=20, alpha=0.7, 
                label='Normal', color='lightblue', density=True)
    axes[0].hist(static_df[static_df['stroke'] == 1]['age'], bins=20, alpha=0.7, 
                label='Stroke', color='lightcoral', density=True)
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Age Distribution by Stroke Status', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Age groups analysis
    age_bins = [18, 30, 40, 50, 60, 70, 90]
    age_labels = ['18-30', '30-40', '40-50', '50-60', '60-70', '70+']
    static_df['age_group'] = pd.cut(static_df['age'], bins=age_bins, labels=age_labels, right=False)
    
    age_stroke_rates = static_df.groupby('age_group')['stroke'].agg(['count', 'sum', 'mean']).reset_index()
    age_stroke_rates['stroke_rate'] = age_stroke_rates['sum'] / age_stroke_rates['count']
    
    bars = axes[1].bar(range(len(age_labels)), age_stroke_rates['stroke_rate'], 
                      color='lightcoral', alpha=0.7)
    axes[1].set_xticks(range(len(age_labels)))
    axes[1].set_xticklabels(age_labels)
    axes[1].set_xlabel('Age Group')
    axes[1].set_ylabel('Stroke Rate')
    axes[1].set_title('Stroke Rate by Age Group', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.1%}', ha='center', va='bottom')
    
    # Combined risk factors with age
    risk_combinations = []
    for age_group in age_labels:
        age_data = static_df[static_df['age_group'] == age_group]
        high_risk = age_data[(age_data['hypertension'] == 1) | (age_data['heart_disease'] == 1)]
        if len(high_risk) > 0:
            stroke_rate = high_risk['stroke'].mean()
            risk_combinations.append((age_group, 'High Risk', stroke_rate, len(high_risk)))
        
        low_risk = age_data[(age_data['hypertension'] == 0) & (age_data['heart_disease'] == 0)]
        if len(low_risk) > 0:
            stroke_rate = low_risk['stroke'].mean()
            risk_combinations.append((age_group, 'Low Risk', stroke_rate, len(low_risk)))
    
    risk_df = pd.DataFrame(risk_combinations, columns=['age_group', 'risk_level', 'stroke_rate', 'count'])
    
    # Pivot for grouped bar chart
    pivot_df = risk_df.pivot(index='age_group', columns='risk_level', values='stroke_rate').fillna(0)
    
    x = np.arange(len(age_labels))
    width = 0.35
    
    if 'High Risk' in pivot_df.columns:
        axes[2].bar(x - width/2, pivot_df['High Risk'], width, label='High Risk', color='red', alpha=0.7)
    if 'Low Risk' in pivot_df.columns:
        axes[2].bar(x + width/2, pivot_df['Low Risk'], width, label='Low Risk', color='blue', alpha=0.7)
    
    axes[2].set_xlabel('Age Group')
    axes[2].set_ylabel('Stroke Rate')
    axes[2].set_title('Stroke Rate by Age Group and Risk Level', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(age_labels)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('age_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_sensor_thresholds_analysis(ts_data, static_df, ts_features):
    """Analyze sensor value thresholds that indicate stroke risk"""
    print("Creating sensor thresholds analysis...")
    
    stroke_indices = static_df['stroke'] == 1
    normal_indices = static_df['stroke'] == 0
    
    # Calculate percentiles for each sensor
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Sensor Value Distributions: Normal vs Stroke Patients', fontsize=16, fontweight='bold')
    
    threshold_data = []
    
    for i, feature in enumerate(ts_features):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Get data for this sensor
        normal_data = ts_data[normal_indices, :, i].flatten()
        stroke_data = ts_data[stroke_indices, :, i].flatten()
        
        # Calculate percentiles
        normal_percentiles = np.percentile(normal_data, [5, 25, 50, 75, 95])
        stroke_percentiles = np.percentile(stroke_data, [5, 25, 50, 75, 95])
        
        # Create box plot comparison
        data_to_plot = [normal_data, stroke_data]
        box_plot = ax.boxplot(data_to_plot, labels=['Normal', 'Stroke'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_title(f'{feature}', fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add threshold lines
        critical_threshold = np.percentile(stroke_data, 75)  # 75th percentile of stroke patients
        ax.axhline(y=critical_threshold, color='red', linestyle='--', alpha=0.7, 
                  label=f'Critical: {critical_threshold:.2f}')
        
        warning_threshold = np.percentile(stroke_data, 50)  # Median of stroke patients
        ax.axhline(y=warning_threshold, color='orange', linestyle='--', alpha=0.7,
                  label=f'Warning: {warning_threshold:.2f}')
        
        ax.legend(fontsize=8)
        
        # Store threshold data
        threshold_data.append({
            'feature': feature,
            'normal_median': np.median(normal_data),
            'stroke_median': np.median(stroke_data),
            'warning_threshold': warning_threshold,
            'critical_threshold': critical_threshold,
            'normal_95th': normal_percentiles[4],
            'stroke_5th': stroke_percentiles[0]
        })
    
    plt.tight_layout()
    plt.savefig('sensor_thresholds_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create threshold summary table
    threshold_df = pd.DataFrame(threshold_data)
    
    plt.figure(figsize=(12, 8))
    plt.axis('tight')
    plt.axis('off')
    
    table_data = []
    for _, row in threshold_df.iterrows():
        table_data.append([
            row['feature'],
            f"{row['normal_median']:.2f}",
            f"{row['stroke_median']:.2f}",
            f"{row['warning_threshold']:.2f}",
            f"{row['critical_threshold']:.2f}"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Sensor', 'Normal Median', 'Stroke Median', 'Warning Threshold', 'Critical Threshold'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the headers
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Sensor Threshold Summary for Stroke Detection', 
              fontweight='bold', fontsize=16, pad=20)
    plt.savefig('sensor_thresholds_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return threshold_df

def main():
    """Main analysis function"""
    print("🚀 Starting comprehensive stroke classification analysis...")
    
    # Load data
    static_df, ts_data, static_features, ts_features = load_and_prepare_data()
    
    print(f"Dataset overview:")
    print(f"- Total patients: {len(static_df)}")
    print(f"- Stroke patients: {static_df['stroke'].sum()} ({static_df['stroke'].mean():.1%})")
    print(f"- Normal patients: {(static_df['stroke'] == 0).sum()} ({(static_df['stroke'] == 0).mean():.1%})")
    print(f"- Time series shape: {ts_data.shape}")
    print()
    
    # Analyze static features
    stroke_patients, normal_patients = analyze_static_features(static_df, static_features)
    
    # Analyze time series features
    stats_features, feature_names = analyze_time_series_features(ts_data, static_df, ts_features)
    
    # Create feature importance analysis
    feature_importance_df = create_feature_importance_analysis(static_df, ts_data, static_features, ts_features)
    
    # Age analysis
    create_age_analysis(static_df)
    
    # Sensor thresholds analysis
    threshold_df = create_sensor_thresholds_analysis(ts_data, static_df, ts_features)
    
    # Print summary
    print("📊 Analysis Summary:")
    print("=" * 50)
    print("\n🔝 Top 5 Most Important Features:")
    top_5 = feature_importance_df.tail(5)
    for _, row in top_5.iterrows():
        feature_type = "Static" if row['feature'] in static_features else "Time Series"
        print(f"   {row['feature']}: {row['importance']:.3f} ({feature_type})")
    
    print(f"\n💡 Key Insights:")
    print(f"   - Age is {'the most' if feature_importance_df.iloc[-1]['feature'] == 'age' else 'a very'} important factor")
    print(f"   - Static features account for {len([f for f in top_5['feature'] if f in static_features])} of top 5 features")
    print(f"   - Time series features provide valuable dynamic information")
    
    print(f"\n📈 Files Generated:")
    plots = [
        'static_features_analysis.png',
        'correlation_heatmap.png', 
        'timeseries_features_analysis.png',
        'feature_importance_analysis.png',
        'top_features_analysis.png',
        'age_analysis.png',
        'sensor_thresholds_analysis.png',
        'sensor_thresholds_table.png'
    ]
    for plot in plots:
        print(f"   ✅ {plot}")
    
    print(f"\n🎯 Recommendations for Real-Time Monitoring:")
    print(f"   - Focus on top 5 features for immediate alerts")
    print(f"   - Implement graduated thresholds (warning → critical)")
    print(f"   - Consider age-specific thresholds")
    print(f"   - Monitor trend changes in physiological signals")

if __name__ == "__main__":
    main()