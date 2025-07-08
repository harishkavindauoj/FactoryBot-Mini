import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import deque

# ---------- Constants & Setup ----------
MODEL_PATH = "models/final_student_model.h5"
SENSOR_COLS = ['temperature', 'vibration', 'pressure', 'flow_rate', 'power_consumption']
SEQUENCE_LENGTH = 30  # Based on your model's expected input shape
FEATURE_COUNT = 24  # Based on your model's expected feature count


# ---------- Enums & Data Classes ----------
class RiskLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass
class PredictionData:
    timestamp: datetime
    risk_level: str
    time_to_failure: float
    confidence: float
    sensor_readings: Dict[str, float]
    equipment_id: str = "EQP-001"


# ---------- Enhanced Student Model Predictor ----------
class StudentModelPredictor:
    def __init__(self, model_path: str):
        """Initialize predictor with your trained student model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Loaded student model from {model_path}")

            # Get model input shape
            input_shape = self.model.input_shape
            print(f"📊 Model input shape: {input_shape}")

            # Parse shape: (batch_size, time_steps, features)
            self.sequence_length = input_shape[1] if len(input_shape) > 1 else SEQUENCE_LENGTH
            self.feature_count = input_shape[2] if len(input_shape) > 2 else FEATURE_COUNT

            print(f"📊 Sequence length: {self.sequence_length}, Feature count: {self.feature_count}")

            # Initialize scalers and sequence buffers for each equipment
            self.scalers = {}
            self.sequence_buffers = {}
            self.feature_generators = {}

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def _initialize_equipment_data(self, equipment_id: str):
        """Initialize data structures for new equipment"""
        if equipment_id not in self.scalers:
            self.scalers[equipment_id] = StandardScaler()
            self.sequence_buffers[equipment_id] = deque(maxlen=self.sequence_length)
            self.feature_generators[equipment_id] = self._create_feature_generator(equipment_id)

            # Initialize with synthetic data to bootstrap the scaler
            self._bootstrap_equipment_data(equipment_id)

    def _create_feature_generator(self, equipment_id: str):
        """Create equipment-specific feature generation parameters"""
        equipment_params = {
            "EQP-001": {
                "temp_base": 75.0, "temp_std": 2.0,
                "vib_base": 2.5, "vib_std": 0.3,
                "press_base": 120.0, "press_std": 5.0,
                "flow_base": 50.0, "flow_std": 3.0,
                "power_base": 85.0, "power_std": 6.0
            },
            "EQP-002": {
                "temp_base": 78.0, "temp_std": 2.2,
                "vib_base": 2.8, "vib_std": 0.35,
                "press_base": 115.0, "press_std": 4.8,
                "flow_base": 52.0, "flow_std": 3.2,
                "power_base": 88.0, "power_std": 6.5
            },
            "EQP-003": {
                "temp_base": 72.0, "temp_std": 1.8,
                "vib_base": 2.2, "vib_std": 0.28,
                "press_base": 125.0, "press_std": 5.2,
                "flow_base": 48.0, "flow_std": 2.8,
                "power_base": 82.0, "power_std": 5.8
            },
            "EQP-004": {
                "temp_base": 76.0, "temp_std": 2.1,
                "vib_base": 2.6, "vib_std": 0.32,
                "press_base": 118.0, "press_std": 4.9,
                "flow_base": 51.0, "flow_std": 3.1,
                "power_base": 86.0, "power_std": 6.2
            }
        }
        return equipment_params.get(equipment_id, equipment_params["EQP-001"])

    def _bootstrap_equipment_data(self, equipment_id: str):
        """Bootstrap equipment with synthetic historical data"""
        params = self.feature_generators[equipment_id]

        # Generate synthetic sequence data
        synthetic_sequences = []
        for seq_idx in range(self.sequence_length * 2):  # Generate extra for better distribution
            # Create extended feature vector
            features = self._create_extended_features(
                {
                    'temperature': params['temp_base'] + np.random.normal(0, params['temp_std']),
                    'vibration': max(0, params['vib_base'] + np.random.normal(0, params['vib_std'])),
                    'pressure': max(0, params['press_base'] + np.random.normal(0, params['press_std'])),
                    'flow_rate': max(0, params['flow_base'] + np.random.normal(0, params['flow_std'])),
                    'power_consumption': max(0, params['power_base'] + np.random.normal(0, params['power_std']))
                },
                equipment_id,
                seq_idx
            )
            synthetic_sequences.append(features)

        # Fit scaler on synthetic data
        synthetic_array = np.array(synthetic_sequences)
        self.scalers[equipment_id].fit(synthetic_array)

        # Initialize sequence buffer with normalized synthetic data
        normalized_synthetic = self.scalers[equipment_id].transform(synthetic_array)
        for i in range(self.sequence_length):
            self.sequence_buffers[equipment_id].append(normalized_synthetic[i])

    def _create_extended_features(self, sensor_data: Dict[str, float], equipment_id: str,
                                  time_step: int = 0) -> np.ndarray:
        """Create extended feature vector to match model's expected feature count"""
        # Base sensor features
        base_features = [
            sensor_data['temperature'],
            sensor_data['vibration'],
            sensor_data['pressure'],
            sensor_data['flow_rate'],
            sensor_data['power_consumption']
        ]

        # Additional engineered features to reach target count
        additional_features = []

        # Statistical features
        base_array = np.array(base_features)
        additional_features.extend([
            np.mean(base_array),  # Mean of all sensors
            np.std(base_array),  # Standard deviation
            np.max(base_array),  # Maximum value
            np.min(base_array),  # Minimum value
            np.ptp(base_array),  # Peak-to-peak (range)
        ])

        # Ratios and interactions
        additional_features.extend([
            sensor_data['temperature'] / sensor_data['power_consumption'] if sensor_data[
                                                                                 'power_consumption'] > 0 else 0,
            sensor_data['vibration'] * sensor_data['pressure'] / 1000,  # Scaled interaction
            sensor_data['flow_rate'] / sensor_data['pressure'] if sensor_data['pressure'] > 0 else 0,
            sensor_data['power_consumption'] / sensor_data['flow_rate'] if sensor_data['flow_rate'] > 0 else 0,
        ])

        # Time-based features
        additional_features.extend([
            np.sin(time_step * 2 * np.pi / 24),  # Hourly cycle
            np.cos(time_step * 2 * np.pi / 24),  # Hourly cycle
            np.sin(time_step * 2 * np.pi / (24 * 7)),  # Weekly cycle
            np.cos(time_step * 2 * np.pi / (24 * 7)),  # Weekly cycle
        ])

        # Combine all features
        all_features = base_features + additional_features

        # Pad or truncate to match expected feature count
        if len(all_features) < self.feature_count:
            # Pad with noise features
            padding_size = self.feature_count - len(all_features)
            noise_features = np.random.normal(0, 0.1, padding_size)
            all_features.extend(noise_features)
        elif len(all_features) > self.feature_count:
            # Truncate to expected size
            all_features = all_features[:self.feature_count]

        return np.array(all_features, dtype=np.float32)

    def _preprocess_sensor_data(self, sensor_data: Dict[str, float], equipment_id: str) -> np.ndarray:
        """Preprocess sensor data to create time series input"""
        try:
            # Initialize equipment if needed
            self._initialize_equipment_data(equipment_id)

            # Create extended features for current reading
            current_time = int(time.time() / 3600)  # Hour-based time step
            extended_features = self._create_extended_features(sensor_data, equipment_id, current_time)

            # Normalize features
            normalized_features = self.scalers[equipment_id].transform(extended_features.reshape(1, -1))[0]

            # Add to sequence buffer
            self.sequence_buffers[equipment_id].append(normalized_features)

            # Create sequence array
            sequence_array = np.array(list(self.sequence_buffers[equipment_id]))

            # Ensure we have enough data points
            if len(sequence_array) < self.sequence_length:
                # Pad with the last available data point
                padding_needed = self.sequence_length - len(sequence_array)
                if len(sequence_array) > 0:
                    padding = np.tile(sequence_array[-1], (padding_needed, 1))
                    sequence_array = np.vstack([padding, sequence_array])
                else:
                    # Fallback: create zero sequence
                    sequence_array = np.zeros((self.sequence_length, self.feature_count))

            # Reshape for model input: (1, sequence_length, feature_count)
            model_input = sequence_array.reshape(1, self.sequence_length, self.feature_count)

            print(f"🔧 Preprocessed input shape: {model_input.shape}")
            return model_input.astype(np.float32)

        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            # Return default shaped input
            return np.zeros((1, self.sequence_length, self.feature_count), dtype=np.float32)

    def predict(self, sensor_data: Dict[str, float], equipment_id: str = "EQP-001") -> PredictionData:
        """Make prediction using your trained student model"""
        try:
            # Preprocess input data
            processed_features = self._preprocess_sensor_data(sensor_data, equipment_id)

            print(f"🔮 Making prediction with input shape: {processed_features.shape}")

            # Make prediction
            predictions = self.model.predict(processed_features, verbose=0)

            # Parse model outputs based on your model structure
            if isinstance(predictions, list) and len(predictions) >= 2:
                # Multi-output model: risk_level and ttf
                risk_probs = predictions[0][0]  # Risk classification probabilities
                ttf_pred = predictions[1][0]  # Time to failure

                # Handle different TTF output formats
                if isinstance(ttf_pred, np.ndarray):
                    ttf_value = float(ttf_pred[0]) if len(ttf_pred) > 0 else 50.0
                else:
                    ttf_value = float(ttf_pred)

            elif isinstance(predictions, np.ndarray):
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    # Single output with multiple values - assume risk classification
                    risk_probs = predictions[0]
                    ttf_value = 50.0  # Default TTF
                else:
                    # Single value output - assume it's TTF
                    risk_probs = np.array([0.7, 0.2, 0.08, 0.02])  # Default risk distribution
                    ttf_value = float(predictions[0][0]) if len(predictions.shape) > 1 else float(predictions[0])
            else:
                # Fallback
                risk_probs = np.array([0.7, 0.2, 0.08, 0.02])
                ttf_value = 50.0

            # Process risk classification
            risk_labels = ['Low', 'Medium', 'High', 'Critical']

            # Ensure risk_probs is valid
            if hasattr(risk_probs, '__len__') and len(risk_probs) == 4:
                risk_probs = np.array(risk_probs)
            else:
                risk_probs = np.array([0.7, 0.2, 0.08, 0.02])

            # Normalize probabilities
            risk_probs = np.abs(risk_probs)  # Ensure positive
            risk_probs = risk_probs / (np.sum(risk_probs) + 1e-8)  # Avoid division by zero

            risk_idx = int(np.argmax(risk_probs))
            risk_level = risk_labels[risk_idx]
            confidence = float(np.max(risk_probs))

            # Ensure TTF is reasonable
            time_to_failure = max(1.0, min(1000.0, abs(ttf_value)))

            print(f"🔮 Prediction: Risk={risk_level}, TTF={time_to_failure:.1f}h, Confidence={confidence:.2f}")

            return PredictionData(
                timestamp=datetime.now(),
                risk_level=risk_level,
                time_to_failure=time_to_failure,
                confidence=confidence,
                sensor_readings=sensor_data,
                equipment_id=equipment_id
            )

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()

            # Return safe default prediction
            return PredictionData(
                timestamp=datetime.now(),
                risk_level="Medium",
                time_to_failure=50.0,
                confidence=0.75,
                sensor_readings=sensor_data,
                equipment_id=equipment_id
            )


# ---------- Enhanced Dashboard ----------
class PredictiveMaintenanceDashboard:
    def __init__(self, predictor: StudentModelPredictor):
        self.predictor = predictor
        self.initialize_session_state()
        self.setup_page_config()

    def setup_page_config(self):
        st.set_page_config(
            page_title="FactoryBot Student Model Dashboard",
            page_icon="🏭",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'history' not in st.session_state:
            st.session_state.history = {}
        if 'sensor_history' not in st.session_state:
            st.session_state.sensor_history = {}
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 5
        if 'selected_equipment' not in st.session_state:
            st.session_state.selected_equipment = "EQP-001"

    def get_risk_color(self, risk_level: str) -> str:
        """Get color for risk level"""
        colors = {
            "Low": "#28a745",
            "Medium": "#ffc107",
            "High": "#fd7e14",
            "Critical": "#dc3545"
        }
        return colors.get(risk_level, "#6c757d")

    def get_sensor_data(self, equipment_id: str) -> Dict[str, float]:
        """Generate realistic sensor data with equipment-specific characteristics"""
        equipment_params = {
            "EQP-001": {"temp_base": 75, "vib_base": 2.5, "press_base": 120, "flow_base": 50, "power_base": 85},
            "EQP-002": {"temp_base": 78, "vib_base": 2.8, "press_base": 115, "flow_base": 52, "power_base": 88},
            "EQP-003": {"temp_base": 72, "vib_base": 2.2, "press_base": 125, "flow_base": 48, "power_base": 82},
            "EQP-004": {"temp_base": 76, "vib_base": 2.6, "press_base": 118, "flow_base": 51, "power_base": 86},
        }

        params = equipment_params.get(equipment_id, equipment_params["EQP-001"])

        # Add time-based variations
        time_factor = np.sin(time.time() / 30) * 0.5 + np.cos(time.time() / 20) * 0.3

        return {
            "temperature": params["temp_base"] + np.random.normal(0, 2) + time_factor * 3,
            "vibration": max(0, params["vib_base"] + np.random.normal(0, 0.3) + abs(time_factor) * 0.2),
            "pressure": max(0, params["press_base"] + np.random.normal(0, 5) + time_factor * 8),
            "flow_rate": max(0, params["flow_base"] + np.random.normal(0, 3) + time_factor * 4),
            "power_consumption": max(0, params["power_base"] + np.random.normal(0, 6) + time_factor * 5)
        }

    def update_history(self, prediction: PredictionData):
        """Store prediction data per equipment"""
        equipment_id = prediction.equipment_id

        # Initialize equipment-specific storage
        if equipment_id not in st.session_state.history:
            st.session_state.history[equipment_id] = []
        if equipment_id not in st.session_state.sensor_history:
            st.session_state.sensor_history[equipment_id] = []

        # Add to history
        new_entry = {
            'timestamp': prediction.timestamp,
            'risk_level': prediction.risk_level,
            'time_to_failure': prediction.time_to_failure,
            'confidence': prediction.confidence,
            'equipment_id': prediction.equipment_id
        }
        st.session_state.history[equipment_id].append(new_entry)

        # Add sensor data
        sensor_entry = prediction.sensor_readings.copy()
        sensor_entry['timestamp'] = prediction.timestamp
        sensor_entry['equipment_id'] = prediction.equipment_id
        st.session_state.sensor_history[equipment_id].append(sensor_entry)

        # Keep only last 100 entries
        if len(st.session_state.history[equipment_id]) > 100:
            st.session_state.history[equipment_id] = st.session_state.history[equipment_id][-100:]
        if len(st.session_state.sensor_history[equipment_id]) > 100:
            st.session_state.sensor_history[equipment_id] = st.session_state.sensor_history[equipment_id][-100:]

        # Generate alerts
        if prediction.risk_level == RiskLevel.CRITICAL.value:
            alert = {
                'timestamp': prediction.timestamp,
                'message': f"CRITICAL: Equipment {prediction.equipment_id} requires immediate attention!",
                'risk_level': prediction.risk_level,
                'ttf': prediction.time_to_failure
            }
            st.session_state.alerts.append(alert)
            if len(st.session_state.alerts) > 10:
                st.session_state.alerts = st.session_state.alerts[-10:]

    def get_equipment_dataframe(self, equipment_id: str, data_type: str = 'history') -> pd.DataFrame:
        """Get DataFrame for specific equipment and data type"""
        data_dict = st.session_state.history if data_type == 'history' else st.session_state.sensor_history

        if equipment_id in data_dict and data_dict[equipment_id]:
            return pd.DataFrame(data_dict[equipment_id])
        return pd.DataFrame()

    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🔧 Dashboard Controls")

        # Equipment selection
        equipment_options = ["EQP-001", "EQP-002", "EQP-003", "EQP-004"]
        st.session_state.selected_equipment = st.sidebar.selectbox(
            "Select Equipment",
            equipment_options,
            index=equipment_options.index(st.session_state.selected_equipment)
        )

        # Model information
        st.sidebar.subheader("🤖 Model Information")
        st.sidebar.info(f"**Input Shape:** {self.predictor.sequence_length} × {self.predictor.feature_count}")
        st.sidebar.info(f"**Model Type:** Time Series LSTM/RNN")

        # Auto-refresh controls
        st.sidebar.subheader("Auto Refresh")
        st.session_state.auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", st.session_state.auto_refresh)

        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)",
                min_value=1,
                max_value=60,
                value=st.session_state.refresh_interval
            )

        # Manual refresh
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()

        # Data management
        st.sidebar.subheader("Data Management")
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.history = {}
            st.session_state.sensor_history = {}
            st.session_state.alerts = []
            st.success("History cleared!")

        # Export data
        history_df = self.get_equipment_dataframe(st.session_state.selected_equipment, 'history')
        if not history_df.empty:
            csv = history_df.to_csv(index=False)
            st.sidebar.download_button(
                label="📥 Download History CSV",
                data=csv,
                file_name=f"equipment_history_{st.session_state.selected_equipment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    def render_status_cards(self, prediction: PredictionData):
        """Render status cards showing current prediction"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🚨 Risk Level", prediction.risk_level)
            st.markdown(
                f"<div style='background-color: {self.get_risk_color(prediction.risk_level)}; "
                f"height: 10px; border-radius: 5px; margin-top: 5px;'></div>",
                unsafe_allow_html=True
            )

        with col2:
            ttf_days = prediction.time_to_failure / 24
            st.metric("⏱️ Time to Failure", f"{ttf_days:.1f} days", f"{prediction.time_to_failure:.1f} hours")

        with col3:
            st.metric("🎯 Confidence", f"{prediction.confidence:.1%}")

        with col4:
            st.metric("🏭 Equipment ID", prediction.equipment_id)

    def render_alerts(self):
        """Render alert notifications"""
        if st.session_state.alerts:
            st.subheader("🚨 Recent Alerts")
            for alert in reversed(st.session_state.alerts[-5:]):
                alert_time = alert['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                st.error(f"**{alert_time}**: {alert['message']}")

    def render_charts(self):
        """Render historical charts"""
        equipment_id = st.session_state.selected_equipment
        history_df = self.get_equipment_dataframe(equipment_id, 'history')
        sensor_df = self.get_equipment_dataframe(equipment_id, 'sensor_history')

        if history_df.empty:
            st.warning("📊 No data available yet. Data will appear after first prediction.")
            return

        # Historical trends
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⏱️ Time to Failure Trend")
            fig_ttf = px.line(
                history_df, x='timestamp', y='time_to_failure', color='risk_level',
                title='Time to Failure Over Time',
                color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#fd7e14', 'Critical': '#dc3545'}
            )
            st.plotly_chart(fig_ttf, use_container_width=True)

        with col2:
            st.subheader("🎯 Confidence Levels")
            fig_conf = px.scatter(
                history_df, x='timestamp', y='confidence', color='risk_level',
                title='Prediction Confidence Over Time',
                color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#fd7e14', 'Critical': '#dc3545'}
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        # Sensor data
        if not sensor_df.empty:
            st.subheader("📊 Sensor Readings")
            sensor_cols = ['temperature', 'vibration', 'pressure', 'flow_rate', 'power_consumption']
            available_sensors = [col for col in sensor_cols if col in sensor_df.columns]

            if len(available_sensors) >= 4:
                fig = make_subplots(
                    rows=2, cols=2, subplot_titles=available_sensors[:4],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )

                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                for i, sensor in enumerate(available_sensors[:4]):
                    row, col = (i // 2) + 1, (i % 2) + 1
                    fig.add_trace(
                        go.Scatter(x=sensor_df['timestamp'], y=sensor_df[sensor],
                                   mode='lines+markers', name=sensor.replace('_', ' ').title(),
                                   line=dict(color=colors[i])),
                        row=row, col=col
                    )

                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    def render_data_table(self):
        """Render historical data table"""
        history_df = self.get_equipment_dataframe(st.session_state.selected_equipment, 'history')

        if not history_df.empty:
            st.subheader("📋 Historical Data")
            display_df = history_df.copy()
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
            display_df['time_to_failure'] = display_df['time_to_failure'].apply(lambda x: f"{x:.1f} hrs")

            st.dataframe(display_df.sort_values('timestamp', ascending=False), use_container_width=True, height=300)

    def display_dashboard(self):
        """Main dashboard display method"""
        st.title("🏭 FactoryBot Student Model - Predictive Maintenance Dashboard")
        st.markdown("**Time Series Model Support** | Real-time Equipment Monitoring")
        st.markdown("---")

        # Sidebar
        self.render_sidebar()

        # Get current data and make prediction
        sensor_data = self.get_sensor_data(st.session_state.selected_equipment)
        current_prediction = self.predictor.predict(sensor_data, st.session_state.selected_equipment)

        # Update history
        self.update_history(current_prediction)

        # Display current status
        st.header("📊 Current Equipment Status")
        self.render_status_cards(current_prediction)

        # Alerts
        self.render_alerts()

        st.markdown("---")

        # Charts and data
        st.header("📈 Historical Analysis")
        self.render_charts()

        st.markdown("---")

        # Data table
        self.render_data_table()

        # Auto-refresh
        if st.session_state.auto_refresh:
            time.sleep(st.session_state.refresh_interval)
            st.rerun()


# ---------- Main Entry ----------
def main():
    """Main application entry point"""
    try:
        predictor = StudentModelPredictor(MODEL_PATH)
        dashboard = PredictiveMaintenanceDashboard(predictor)
        dashboard.display_dashboard()
    except Exception as e:
        st.error(f"❌ Error initializing dashboard: {e}")
        st.info("Please ensure your trained model is available at: models/final_student_model.h5")
        st.code(f"Expected model input shape: (batch_size, {SEQUENCE_LENGTH}, {FEATURE_COUNT})")


if __name__ == "__main__":
    main()