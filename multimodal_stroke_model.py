import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import shap
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# GPU Setup Verification
print("=== GPU Setup Check ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU memory: {gpu_props.total_memory / 1024**3:.1f} GB")
    print(f"Compute capability: {gpu_props.major}.{gpu_props.minor}")
else:
    print("WARNING: CUDA not available, using CPU")
print("========================\n")

# Load synthetic data
static_df = pd.read_csv('synthetic_static_data.csv')
ts_data = np.load('synthetic_ts_data.npy')
labels = static_df['stroke'].values

# Preprocess
scaler = MinMaxScaler()
static_scaled = scaler.fit_transform(static_df.drop('stroke', axis=1))
ts_scaled = np.array([scaler.fit_transform(ts) for ts in ts_data])

# Check dimensions and fix model initialization
static_features = static_df.drop('stroke', axis=1)
print(f"Static features shape: {static_features.shape}")
print(f"Static feature names: {static_features.columns.tolist()}")

# Train-test split
X_static_train, X_static_test, X_ts_train, X_ts_test, y_train, y_test = train_test_split(
    static_scaled, ts_scaled, labels, test_size=0.2, stratify=labels, random_state=42
)

# Define LSTM with Attention
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h):
        attention_weights = self.softmax(self.attention(h))
        context = torch.sum(h * attention_weights, dim=1)
        return context, attention_weights

class StrokeLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=32, static_dim=6):  # Reduced for Quadro M1200
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)  # Reduced layers
        self.attention = Attention(hidden_dim)
        self.fc_static = nn.Linear(static_dim, 32)  # Reduced size
        self.fc = nn.Linear(hidden_dim + 32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_ts, x_static):
        lstm_out, _ = self.lstm(x_ts)
        context, attn_weights = self.attention(lstm_out)
        static_out = self.fc_static(x_static)
        combined = torch.cat((context, static_out), dim=1)
        output = self.sigmoid(self.fc(combined))
        return output, attn_weights

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

lstm_model = StrokeLSTM(static_dim=static_features.shape[1]).to(device)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train RF on static data
rf_model.fit(X_static_train, y_train)

# Create DataLoader for batch processing
batch_size = 16  # Smaller batch for Quadro M1200
train_dataset = TensorDataset(
    torch.FloatTensor(X_ts_train), 
    torch.FloatTensor(X_static_train), 
    torch.FloatTensor(y_train)
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train LSTM with batches
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
criterion = nn.BCELoss()

print("Starting LSTM training...")
for epoch in range(10):  # Reduced epochs for testing
    lstm_model.train()
    total_loss = 0
    for batch_idx, (batch_ts, batch_static, batch_y) in enumerate(train_loader):
        batch_ts = batch_ts.to(device)
        batch_static = batch_static.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs, _ = lstm_model(batch_ts, batch_static)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Clear cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

# Evaluate with batches
print("Evaluating model...")
lstm_model.eval()
test_dataset = TensorDataset(
    torch.FloatTensor(X_ts_test), 
    torch.FloatTensor(X_static_test)
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

lstm_predictions = []
with torch.no_grad():
    for batch_ts, batch_static in test_loader:
        batch_ts = batch_ts.to(device)
        batch_static = batch_static.to(device)
        outputs, _ = lstm_model(batch_ts, batch_static)
        lstm_predictions.extend(outputs.squeeze().cpu().numpy())

lstm_pred = np.array(lstm_predictions)
rf_pred = rf_model.predict_proba(X_static_test)[:, 1]
ensemble_pred = 0.7 * lstm_pred + 0.3 * rf_pred
y_pred = (ensemble_pred > 0.5).astype(int)

# Metrics
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'AUC: {roc_auc_score(y_test, ensemble_pred):.4f}')

# Clear GPU memory before SHAP
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Explainability - save plots to files
print("Computing SHAP values...")
try:
    explainer = shap.KernelExplainer(rf_model.predict_proba, X_static_train[:50])  # Smaller sample
    shap_values = explainer.shap_values(X_static_test[:25])  # Even smaller test sample
    
    # Save SHAP summary plot to file
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_static_test[:25], 
                     feature_names=static_df.drop('stroke', axis=1).columns,
                     show=False)  # Don't show, just save
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved as 'shap_summary_plot.png'")
    
    # Save feature importance plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_static_test[:25], 
                     feature_names=static_df.drop('stroke', axis=1).columns,
                     plot_type="bar", show=False)
    plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP feature importance plot saved as 'shap_feature_importance.png'")
    
except Exception as e:
    print(f"SHAP visualization failed: {e}")
    print("Model training completed successfully without SHAP plots")

print("\nModel training and evaluation completed!")