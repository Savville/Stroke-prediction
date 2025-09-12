import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Simplified LSTM for edge deployment
class EdgeStrokeLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=16, static_dim=6):  # Even smaller for edge
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc_static = nn.Linear(static_dim, 16)
        self.fc = nn.Linear(hidden_dim + 16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_ts, x_static):
        # Simplified - no attention for faster edge inference
        lstm_out, (h_n, _) = self.lstm(x_ts)
        context = h_n[-1]  # Use last hidden state only
        static_out = self.fc_static(x_static)
        combined = torch.cat((context, static_out), dim=1)
        output = self.sigmoid(self.fc(combined))
        return output

def train_edge_model():
    """Train the simplified edge model"""
    # Load data (same as your main script)
    static_df = pd.read_csv('synthetic_static_data.csv')
    ts_data = np.load('synthetic_ts_data.npy')
    labels = static_df['stroke'].values
    
    # Preprocess
    scaler_static = MinMaxScaler()
    scaler_ts = MinMaxScaler()
    
    static_scaled = scaler_static.fit_transform(static_df.drop('stroke', axis=1))
    
    # For time series, we need to scale each feature across all samples and timesteps
    # Reshape to (samples * timesteps, features) for proper scaling
    ts_reshaped = ts_data.reshape(-1, ts_data.shape[-1])
    ts_scaled_reshaped = scaler_ts.fit_transform(ts_reshaped)
    ts_scaled = ts_scaled_reshaped.reshape(ts_data.shape)
    
    # Train simplified model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    edge_model = EdgeStrokeLSTM(static_dim=static_scaled.shape[1]).to(device)
    
    # Quick training
    optimizer = torch.optim.Adam(edge_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Simple training loop
    edge_model.train()
    for epoch in range(5):  # Quick training for demo
        epoch_loss = 0
        batch_count = 0
        
        for i in range(0, len(ts_scaled), 32):
            end_idx = min(i + 32, len(ts_scaled))
            
            batch_ts = torch.FloatTensor(ts_scaled[i:end_idx]).to(device)
            batch_static = torch.FloatTensor(static_scaled[i:end_idx]).to(device)
            batch_y = torch.FloatTensor(labels[i:end_idx]).to(device)
            
            optimizer.zero_grad()
            outputs = edge_model(batch_ts, batch_static)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # Clear GPU memory periodically
            if batch_count % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / batch_count
        print(f'Edge model Epoch {epoch+1} completed - Loss: {avg_loss:.4f}')
    
    return edge_model, scaler_static, scaler_ts, device

def export_to_onnx(model, scaler_static, scaler_ts, device):
    """Export model to ONNX for edge deployment"""
    model.eval()
    
    # Get the actual static dimension from the model
    static_dim = model.fc_static.in_features
    print(f"Model static dimension: {static_dim}")
    
    # Create dummy inputs for export - use batch_size=1 to avoid LSTM warnings
    dummy_ts = torch.randn(1, 300, 9).to(device)
    dummy_static = torch.randn(1, static_dim).to(device)
    
    print("Exporting model to ONNX...")
    
    # Move model to CPU for ONNX export (recommended)
    model_cpu = model.cpu()
    dummy_ts_cpu = dummy_ts.cpu()
    dummy_static_cpu = dummy_static.cpu()
    
    try:
        # Export to ONNX with batch_size=1 to avoid warnings
        torch.onnx.export(
            model_cpu,
            (dummy_ts_cpu, dummy_static_cpu),
            "stroke_edge_model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['time_series', 'static_features'],
            output_names=['stroke_probability'],
            # Remove dynamic_axes to use fixed batch_size=1
        )
        
        # Save scaler parameters with correct MinMaxScaler attributes
        scaler_data = {
            'static_min': scaler_static.data_min_,
            'static_max': scaler_static.data_max_,
            'static_scale': scaler_static.scale_,
            'ts_min': scaler_ts.data_min_,
            'ts_max': scaler_ts.data_max_,
            'ts_scale': scaler_ts.scale_,
            'static_dim': static_dim,
            'feature_names': ['age', 'hypertension', 'heart_disease', 'bmi', 'glucose', 'smoking']
        }
        np.save('scaler_params.npy', scaler_data)
        
        print("✅ Model exported to 'stroke_edge_model.onnx'")
        print("✅ Scaler parameters saved to 'scaler_params.npy'")
        
        # Test the exported model
        test_onnx_model(dummy_ts_cpu, dummy_static_cpu)
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        print("Saving PyTorch model instead...")
        torch.save(model_cpu.state_dict(), 'stroke_edge_model.pth')
        print("✅ PyTorch model saved to 'stroke_edge_model.pth'")
        
        # Still save scaler parameters for fallback
        scaler_data = {
            'static_min': scaler_static.data_min_,
            'static_max': scaler_static.data_max_, 
            'static_scale': scaler_static.scale_,
            'ts_min': scaler_ts.data_min_,
            'ts_max': scaler_ts.data_max_,
            'ts_scale': scaler_ts.scale_,
            'static_dim': static_dim,
            'feature_names': ['age', 'hypertension', 'heart_disease', 'bmi', 'glucose', 'smoking']
        }
        np.save('scaler_params.npy', scaler_data)
        print("✅ Scaler parameters saved to 'scaler_params.npy'")

def test_onnx_model(dummy_ts, dummy_static):
    """Test the exported ONNX model"""
    try:
        print("\nTesting ONNX model...")
        session = ort.InferenceSession("stroke_edge_model.onnx")
        
        # Get input info
        input_info = session.get_inputs()
        print(f"Model inputs: {[inp.name for inp in input_info]}")
        print(f"Input shapes: {[inp.shape for inp in input_info]}")
        
        # Run inference
        result = session.run(
            ['stroke_probability'],
            {
                'time_series': dummy_ts.numpy().astype(np.float32),
                'static_features': dummy_static.numpy().astype(np.float32)
            }
        )
        
        print(f"✅ ONNX model test successful!")
        print(f"   Output shape: {result[0].shape}")
        print(f"   Sample prediction: {result[0][0][0]:.4f}")
        
    except Exception as e:
        print(f"❌ ONNX model test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting edge model training and export...")
    
    try:
        # Train and export edge model
        edge_model, scaler_static, scaler_ts, device = train_edge_model()
        export_to_onnx(edge_model, scaler_static, scaler_ts, device)
        
        print("\n🚀 Edge deployment ready!")
        print("   Files created:")
        print("   - stroke_edge_model.onnx (or stroke_edge_model.pth)")
        print("   - scaler_params.npy")
        print("   Next step: Run 'python real_time_monitor.py'")
        
    except Exception as e:
        print(f"❌ Process failed: {e}")
        import traceback
        traceback.print_exc()