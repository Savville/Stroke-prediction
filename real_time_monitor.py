import numpy as np
import onnxruntime as ort
import time
from collections import deque
import os

class RealTimeStrokeMonitor:
    def __init__(self, model_path="stroke_edge_model.onnx", scaler_path="scaler_params.npy"):
        """Initialize real-time stroke monitoring system"""
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        try:
            # Load ONNX model
            self.session = ort.InferenceSession(model_path)
            print(f"✅ Model loaded: {model_path}")
            
            # Load scaler parameters
            scaler_data = np.load(scaler_path, allow_pickle=True).item()
            
            # Updated to use correct MinMaxScaler attributes
            self.static_min = scaler_data['static_min']
            self.static_max = scaler_data['static_max']
            self.static_scale = scaler_data['static_scale']
            self.ts_min = scaler_data['ts_min']
            self.ts_max = scaler_data['ts_max']
            self.ts_scale = scaler_data['ts_scale']
            
            print(f"✅ Scaler loaded: {scaler_path}")
            
        except Exception as e:
            print(f"❌ Error loading files: {e}")
            print("Please run 'python edge_optimized_model.py' first to generate the required files.")
            raise
        
        # Initialize sliding window buffer
        self.window_size = 300
        self.ts_buffer = deque(maxlen=self.window_size)
        
        # Alert system
        self.alert_threshold = 0.7
        self.consecutive_alerts = 0
        self.alert_confirm_count = 3
        
        print(f"Real-time stroke monitor initialized")
        print(f"Window size: {self.window_size}")
        print(f"Alert threshold: {self.alert_threshold}")
    
    def normalize_static_data(self, data):
        """Normalize static data using MinMaxScaler formula"""
        data = np.array(data)
        return (data - self.static_min) * self.static_scale
    
    def normalize_ts_data(self, data):
        """Normalize time series data using MinMaxScaler formula"""
        data = np.array(data)
        return (data - self.ts_min) * self.ts_scale
    
    def add_sensor_reading(self, hr, sys_bp, dia_bp, spo2, rr, qrs, acc_x, acc_y, acc_z):
        """Add new sensor reading to buffer"""
        sensor_data = [hr, sys_bp, dia_bp, spo2, rr, qrs, acc_x, acc_y, acc_z]
        self.ts_buffer.append(sensor_data)
        
        return {
            'buffer_size': len(self.ts_buffer),
            'ready_for_prediction': len(self.ts_buffer) >= self.window_size
        }
    
    def predict_stroke_risk(self, static_features):
        """Real-time stroke risk prediction"""
        if len(self.ts_buffer) < self.window_size:
            return {
                'status': 'insufficient_data',
                'buffer_progress': f"{len(self.ts_buffer)}/{self.window_size}",
                'stroke_probability': None,
                'alert_level': 'none'
            }
        
        try:
            # Prepare time series input
            ts_input = np.array(list(self.ts_buffer)).reshape(1, self.window_size, 9)
            ts_input = self.normalize_ts_data(ts_input)
            
            # Prepare static input
            static_input = np.array(static_features).reshape(1, -1)
            static_input = self.normalize_static_data(static_input)
            
            # Run inference
            start_time = time.time()
            result = self.session.run(
                ['stroke_probability'],
                {
                    'time_series': ts_input.astype(np.float32),
                    'static_features': static_input.astype(np.float32)
                }
            )
            inference_time = (time.time() - start_time) * 1000
            
            stroke_prob = float(result[0][0][0])
            alert_level = self._determine_alert_level(stroke_prob)
            
            return {
                'status': 'success',
                'stroke_probability': stroke_prob,
                'alert_level': alert_level,
                'inference_time_ms': round(inference_time, 2),
                'consecutive_alerts': self.consecutive_alerts,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'stroke_probability': None,
                'alert_level': 'error'
            }
    
    def _determine_alert_level(self, stroke_prob):
        """Determine alert level based on probability and consecutive alerts"""
        if stroke_prob >= self.alert_threshold:
            self.consecutive_alerts += 1
            if self.consecutive_alerts >= self.alert_confirm_count:
                return 'critical'
            else:
                return 'warning'
        else:
            self.consecutive_alerts = 0
            return 'normal'
    
    def reset_buffer(self):
        """Reset the monitoring buffer"""
        self.ts_buffer.clear()
        self.consecutive_alerts = 0
        print("Monitor buffer reset")

# Rest of the file remains the same...
def demo_real_time_monitoring():
    """Demonstrate real-time stroke monitoring"""
    
    try:
        monitor = RealTimeStrokeMonitor()
    except Exception as e:
        print(f"❌ Failed to initialize monitor: {e}")
        return
    
    # Patient static data (example)
    patient_static = [65, 1, 0, 28.5, 120, 0]  # age, hypertension, heart_disease, bmi, glucose, smoking
    
    print("\nStarting real-time monitoring demo...")
    print("Simulating sensor readings...")
    print("Legend: Normal -> Warning -> Critical")
    print("-" * 50)
    
    # Simulate 350 time steps of sensor data
    for i in range(350):
        # Simulate normal readings initially, then stroke-like readings
        if i < 250:
            # Normal readings
            hr = np.random.normal(70, 5)
            sys_bp = np.random.normal(120, 10)
            dia_bp = np.random.normal(80, 5)
            spo2 = np.random.normal(98, 1)
            rr = np.random.normal(0.8, 0.05)
            qrs = np.random.normal(1, 0.1)
            acc_x = np.random.normal(0, 0.3)
            acc_y = np.random.normal(0, 0.3)
            acc_z = np.random.normal(0, 0.3)
        else:
            # Stroke-like readings
            hr = np.random.normal(80, 10) + (i - 250) * 0.1
            sys_bp = np.random.normal(130, 15) + (i - 250) * 0.2
            dia_bp = np.random.normal(85, 10) + (i - 250) * 0.1
            spo2 = np.random.normal(95, 2) - (i - 250) * 0.01
            rr = np.random.normal(0.8, 0.2) * (1 + np.random.uniform(-0.1, 0.1))
            qrs = np.random.normal(1, 0.2) - (i - 250) * 0.001
            acc_x = np.random.normal(0, 0.5) + np.random.uniform(-0.1, 0.1)
            acc_y = np.random.normal(0, 0.5)
            acc_z = np.random.normal(0, 0.5)
        
        # Add sensor reading
        buffer_status = monitor.add_sensor_reading(hr, sys_bp, dia_bp, spo2, rr, qrs, acc_x, acc_y, acc_z)
        
        # Show buffer filling progress
        if i < 300 and i % 50 == 0:
            print(f"Filling buffer: {buffer_status['buffer_size']}/{monitor.window_size}")
        
        # Predict every 10 readings
        if i % 10 == 0 and buffer_status['ready_for_prediction']:
            result = monitor.predict_stroke_risk(patient_static)
            
            alert_emoji = {
                'normal': '✅',
                'warning': '⚠️ ',
                'critical': '🚨',
                'error': '❌'
            }
            
            emoji = alert_emoji.get(result.get('alert_level', 'none'), '❓')
            
            print(f"{emoji} Time {i:3d}: "
                  f"Prob={result.get('stroke_probability', 0):.3f}, "
                  f"Alert={result.get('alert_level', 'none'):8s}, "
                  f"Time={result.get('inference_time_ms', 0):5.1f}ms")
            
            if result.get('alert_level') == 'critical':
                print("\n" + "="*60)
                print("🚨 CRITICAL ALERT: High stroke risk detected!")
                print("   📞 Contact emergency services immediately!")
                print("   🏥 Stroke probability:", f"{result.get('stroke_probability', 0):.1%}")
                print("="*60)
                break
        
        time.sleep(0.01)
    
    print("\n✅ Real-time monitoring demo completed!")

if __name__ == "__main__":
    demo_real_time_monitoring()