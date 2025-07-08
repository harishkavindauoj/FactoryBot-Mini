# 🏭 FactoryBot-Mini: Predictive Maintenance via Knowledge Distillation

FactoryBot-Mini is an end-to-end predictive maintenance pipeline built on NASA's CMAPSS dataset. It uses a **Gemini LLM teacher** to distill risk and time-to-failure (TTF) labels, which are learned by a compact **student neural network** for deployment.

---

## 📦 Project Structure

```
FactoryBot-Mini/
├── config/           # Configuration for models, prompts, etc.
├── data/             # Raw and processed CMAPSS data
├── models/           # Trained student model & preprocessor
├── notebooks/        # Exploratory analysis and distillation inspection
├── prompts/          # Gemini prompt templates
├── reports/          # Evaluation plots and metrics
├── src/              # Source code: distillation, model, training, eval
├── tests/            # Unit tests
├── run.py            # CLI entrypoint
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo:
```bash
git clone https://github.com/your-org/FactoryBot-Mini.git
cd FactoryBot-Mini
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Add a `.env` file:
```ini
GEMINI_API_KEY=your_google_generative_ai_key
GEMINI_MODEL_NAME=gemini-pro
```

### 4. Download raw CMAPSS data (automated):
The preprocessor will download `train_FD001.txt` on first run.

---

## 🚀 Running the Pipeline

To run the full training-evaluation loop:

```bash
python run.py
```

This will:
- Preprocess CMAPSS data into windowed sensor matrices
- Generate soft labels (risk level + TTF) using Gemini
- Train the student neural network
- Evaluate classification + regression performance

---

## 🧠 Soft Label Modes

There are 3 options for generating LLM-based soft labels:

### 1. ✅ Smart Sampling (Recommended)
Generates Gemini labels on a representative subset (5–10%) and propagates them across the dataset using PCA + nearest neighbor.

```python
config['use_sampling'] = True
config['sample_ratio'] = 0.1
```

**➡️ Fast and high-quality (~20–30 mins)**

### 2. ⚠️ Emergency Fast Mode (Not for Training)
Generates labels for just 200 samples and repeats them across the full dataset.

```python
config['emergency_fast'] = True
```

**⚠️ Use only for debugging, not for training or evaluation.**

### 3. 🔁 Full Distillation (Slowest)
Run Gemini on all 17,000+ samples.

```python
config['use_sampling'] = False
config['emergency_fast'] = False
```

**➡️ Takes 8–12+ hours, best accuracy (but requires quota)**

### Optional: Force Soft Label Regeneration
To force label regeneration even if cached:

```python
config['force_regenerate'] = True
```

---

## 📈 Evaluation

### Evaluation outputs:
- Confusion matrix (`reports/risk_confusion_matrix.png`)
- Scatter plot of predicted vs. true TTF
- Metrics:
  - **Accuracy, Precision, Recall** per risk level
  - **MAE, R², Pearson Correlation** for TTF

---

## 🧪 Testing

To run unit tests:

```bash
pytest tests/
```

---

## 🔧 Configuration

Key configuration options can be found in `config/` directory:
- Model hyperparameters
- Prompt templates for Gemini
- Data preprocessing settings
- Training parameters

---

## 📊 Dataset

This project uses NASA's Commercial Modular Aero-Propulsion System Simulation (CMAPSS) dataset:
- **FD001**: Engine degradation simulation data
- **Features**: 21 sensor measurements
- **Target**: Remaining Useful Life (RUL) prediction

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- NASA for providing the CMAPSS dataset
- Google for the Gemini API
- The open-source community for various ML libraries used in this project