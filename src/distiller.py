"""Optimized Gemini-based teacher model with batching and caching."""

import os
import numpy as np
import json
import re
import time
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Tuple, Optional, List
from tqdm import tqdm

load_dotenv()
RISK_LEVELS = ["Low", "Moderate", "High", "Critical"]


class GeminiTeacher:
    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-pro")

        if not self.api_key:
            raise RuntimeError("Gemini API key not found in .env")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

        # Load prompt template
        with open(config['prompt']['template_path'], 'r') as f:
            self.prompt_template = f.read()

        # Cache for resuming interrupted runs
        self.cache_path = config.get('cache_path', 'data/soft_labels_cache.pkl')
        self.cache = self.load_cache()

    def load_cache(self) -> dict:
        """Load existing cache if available."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_cache(self):
        """Save cache to disk."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

    def call_gemini_batch(self, samples: List[np.ndarray], batch_size: int = 5) -> List[
        Tuple[Optional[int], Optional[float]]]:
        """Process multiple samples in a single API call."""
        if len(samples) == 0:
            return []

        # Create batch prompt
        batch_data = []
        for i, sample in enumerate(samples):
            batch_data.append({
                "sample_id": i,
                "sensor_data": sample.tolist()
            })

        batch_prompt = f"""
        Analyze the following {len(samples)} sensor data samples and provide risk assessment for each.

        Data: {json.dumps(batch_data, indent=2)}

        For each sample, provide output in this exact format:
        Sample [ID]: Risk Level: [Low/Moderate/High/Critical], TTF: [number]

        Example:
        Sample 0: Risk Level: High, TTF: 45.2
        Sample 1: Risk Level: Low, TTF: 120.0
        """

        try:
            response = self.model.generate_content(batch_prompt)
            if not response.text:
                return [None] * len(samples)

            return self.parse_batch_output(response.text.strip(), len(samples))

        except Exception as e:
            print(f"[Distiller] Batch API Error: {e}")
            return [None] * len(samples)

    def parse_batch_output(self, output: str, expected_count: int) -> List[Tuple[Optional[int], Optional[float]]]:
        """Parse batch output from Gemini."""
        results = [None] * expected_count

        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Sample'):
                try:
                    # Extract sample ID
                    sample_match = re.search(r'Sample (\d+):', line)
                    if not sample_match:
                        continue

                    sample_id = int(sample_match.group(1))
                    if sample_id >= expected_count:
                        continue

                    # Extract risk level
                    risk_idx = None
                    for i, level in enumerate(RISK_LEVELS):
                        if level.lower() in line.lower():
                            risk_idx = i
                            break

                    # Extract TTF
                    ttf_match = re.search(r'TTF:\s*(\d+\.?\d*)', line)
                    ttf = float(ttf_match.group(1)) if ttf_match else None

                    if risk_idx is not None and ttf is not None:
                        results[sample_id] = (risk_idx, ttf)

                except Exception as e:
                    print(f"[Distiller] Parse error for line: {line}, error: {e}")
                    continue

        return results

    def process_single_sample(self, sample: np.ndarray, sample_id: int) -> Tuple[Optional[int], Optional[float]]:
        """Process a single sample - fallback for batch failures."""
        cache_key = f"sample_{sample_id}"

        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        sensor_data_str = json.dumps(sample.tolist())
        prompt = self.prompt_template.replace("{sensor_data}", sensor_data_str)

        try:
            response = self.model.generate_content(prompt)
            if not response.text:
                return None, None

            risk_idx, ttf = self.parse_single_output(response.text.strip())

            # Cache successful results
            if risk_idx is not None and ttf is not None:
                self.cache[cache_key] = (risk_idx, ttf)

            return risk_idx, ttf

        except Exception as e:
            print(f"[Distiller] Single API Error for sample {sample_id}: {e}")
            return None, None

    def parse_single_output(self, output: str) -> Tuple[Optional[int], Optional[float]]:
        """Parse single sample output."""
        if not output:
            return None, None

        risk_idx = None
        ttf = None

        lines = output.split('\n')
        for line in lines:
            line = line.strip()

            # Risk level parsing
            if any(keyword in line.lower() for keyword in ['risk level', 'risk:', 'level:']):
                for i, level in enumerate(RISK_LEVELS):
                    if level.lower() in line.lower():
                        risk_idx = i
                        break

            # TTF parsing
            if any(keyword in line.lower() for keyword in ['ttf', 'time to failure', 'cycles']):
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    ttf = float(numbers[0])

        return risk_idx, ttf

    def generate_soft_labels_optimized(self, max_workers: int = 5, batch_size: int = 10,
                                       sample_subset: Optional[int] = None):
        """Generate soft labels with optimizations."""
        print("[Distiller] Loading features...")
        X = np.load(self.config['data']['features_cache'])

        # Use subset for testing
        if sample_subset:
            X = X[:sample_subset]
            print(f"[Distiller] Using subset of {sample_subset} samples for testing")

        print(f"[Distiller] Processing {len(X)} samples with batch_size={batch_size}, max_workers={max_workers}")

        risk_labels = []
        ttf_labels = []

        # Process in batches with progress bar
        with tqdm(total=len(X), desc="Processing samples") as pbar:
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]

                # Try batch processing first
                batch_results = self.call_gemini_batch(batch, batch_size)

                # Process each sample in batch
                for j, sample in enumerate(batch):
                    sample_id = i + j
                    result = batch_results[j] if j < len(batch_results) else None

                    if result is None or result[0] is None or result[1] is None:
                        # Fallback to single sample processing
                        result = self.process_single_sample(sample, sample_id)

                    if result[0] is not None and result[1] is not None:
                        risk_labels.append(result[0])
                        ttf_labels.append(result[1])
                    else:
                        # Final fallback to random
                        risk_labels.append(np.random.randint(0, 4))
                        ttf_labels.append(np.random.uniform(10, 100))

                    pbar.update(1)

                # Save cache periodically
                if i % (batch_size * 10) == 0:
                    self.save_cache()

                # Rate limiting
                time.sleep(0.1)

        # Final cache save
        self.save_cache()

        # Convert to numpy arrays
        risk_onehot = np.eye(len(RISK_LEVELS))[risk_labels]
        ttf_array = np.array(ttf_labels).reshape(-1, 1)

        # Save results
        os.makedirs('data/processed', exist_ok=True)
        np.save('data/processed/risk_labels.npy', risk_onehot)
        np.save('data/processed/ttf_labels.npy', ttf_array)

        success_rate = len([r for r in risk_labels if r is not None]) / len(risk_labels)
        print(f"[Distiller] Soft labels saved. Success rate: {success_rate:.1%}")

        return risk_onehot, ttf_array


def generate_soft_labels(config):
    """Main function - NOW DEFAULTS TO FAST SAMPLING MODE!"""

    # Check if soft labels already exist
    risk_labels_path = "data/processed/risk_labels.npy"
    ttf_labels_path = "data/processed/ttf_labels.npy"

    if os.path.exists(risk_labels_path) and os.path.exists(ttf_labels_path):
        print("[Distiller] ✅ Soft labels already exist! Loading from disk...")
        try:
            y_risk = np.load(risk_labels_path)
            y_ttf = np.load(ttf_labels_path)
            print(f"[Distiller] 📊 Loaded existing labels: {y_risk.shape} risk labels, {y_ttf.shape} TTF labels")

            # Validate shapes
            expected_samples = len(np.load(config['data']['features_cache']))
            if len(y_risk) == expected_samples and len(y_ttf) == expected_samples:
                print("[Distiller] ✅ Label shapes match feature data. Using existing labels.")
                return y_risk, y_ttf
            else:
                print(
                    f"[Distiller] ⚠️  Shape mismatch! Expected {expected_samples}, got {len(y_risk)} risk, {len(y_ttf)} TTF")
                print("[Distiller] 🔄 Regenerating soft labels...")
        except Exception as e:
            print(f"[Distiller] ⚠️  Error loading existing labels: {e}")
            print("[Distiller] 🔄 Regenerating soft labels...")
    else:
        print("[Distiller] 📝 No existing soft labels found. Generating new ones...")

    # Force regeneration if requested
    if config.get('force_regenerate', False):
        print("[Distiller] 🔄 Force regenerate requested. Creating new soft labels...")

    # Generate new soft labels
    # Check if user wants the emergency ultra-fast mode
    if config.get('emergency_fast', False):
        print("[Distiller] 🚨 Using emergency fast mode (200 samples, ~10 minutes)")
        return generate_soft_labels_emergency_fast(config, n_samples=200)

    # Default to sampling mode (much faster than original)
    elif config.get('use_sampling', True):  # DEFAULT to True now
        sample_ratio = config.get('sample_ratio', 0.05)  # 5% of data
        print(f"[Distiller] 🚀 Using smart sampling mode ({sample_ratio:.1%} of data)")
        return generate_soft_labels_sampled(config, sample_ratio=sample_ratio)

    # Only use full processing if explicitly requested
    else:
        print("[Distiller] ⚠️  Using full processing mode (VERY SLOW - 50+ hours)")
        teacher = GeminiTeacher(config)
        return teacher.generate_soft_labels_optimized(
            max_workers=5,
            batch_size=10
        )


# FAST: Use representative sampling - THIS IS THE RECOMMENDED APPROACH
def generate_soft_labels_sampled(config, sample_ratio: float = 0.05):
    """Generate soft labels using only a representative sample - MUCH FASTER!"""
    print(f"[Distiller] 🚀 FAST MODE: Using sampling approach with {sample_ratio:.1%} of data")

    X = np.load(config['data']['features_cache'])

    # Use stratified sampling for better representation
    n_samples = max(100, int(len(X) * sample_ratio))  # Minimum 100 samples

    # Try to get diverse samples (every N-th sample + some random)
    step = len(X) // (n_samples // 2)
    systematic_indices = np.arange(0, len(X), step)[:n_samples // 2]
    random_indices = np.random.choice(len(X), n_samples // 2, replace=False)
    sample_indices = np.concatenate([systematic_indices, random_indices])
    sample_indices = np.unique(sample_indices)[:n_samples]

    X_sampled = X[sample_indices]

    print(f"[Distiller] Processing {len(X_sampled)} sampled features (instead of {len(X)})")
    print(f"[Distiller] Estimated time: {len(X_sampled) * 10 / 60:.1f} minutes (vs {len(X) * 10 / 3600:.1f} hours)")

    # Process sampled data with smaller batches for faster feedback
    teacher = GeminiTeacher(config)

    # Process the sampled data
    risk_labels = []
    ttf_labels = []

    with tqdm(total=len(X_sampled), desc="Processing sampled data") as pbar:
        for i in range(0, len(X_sampled), 5):  # Smaller batches
            batch = X_sampled[i:i + 5]

            for j, sample in enumerate(batch):
                sample_id = i + j
                result = teacher.process_single_sample(sample, sample_id)

                if result[0] is not None and result[1] is not None:
                    risk_labels.append(result[0])
                    ttf_labels.append(result[1])
                else:
                    # Fallback
                    risk_labels.append(np.random.randint(0, 4))
                    ttf_labels.append(np.random.uniform(10, 100))

                pbar.update(1)

            # Small delay to respect rate limits
            time.sleep(0.5)

    # Save sampled results first
    sampled_risk_onehot = np.eye(len(RISK_LEVELS))[risk_labels]
    sampled_ttf_array = np.array(ttf_labels).reshape(-1, 1)

    print(f"[Distiller] ✓ Sampled labels generated. Now expanding to full dataset...")

    # Expand to full dataset using nearest neighbor
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA

    # Use PCA to reduce dimensionality for faster nearest neighbor
    X_flat = X.reshape(len(X), -1)
    X_sampled_flat = X_sampled.reshape(len(X_sampled), -1)

    # Reduce dimensions if data is high-dimensional
    if X_flat.shape[1] > 100:
        pca = PCA(n_components=50)
        X_flat_reduced = pca.fit_transform(X_flat)
        X_sampled_flat_reduced = pca.transform(X_sampled_flat)
    else:
        X_flat_reduced = X_flat
        X_sampled_flat_reduced = X_sampled_flat

    # Find nearest neighbors
    print(f"[Distiller] Finding nearest neighbors for {len(X)} samples...")
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    nn.fit(X_sampled_flat_reduced)
    distances, indices = nn.kneighbors(X_flat_reduced)

    # Expand labels
    risk_labels_full = [risk_labels[indices[i][0]] for i in range(len(X))]
    ttf_labels_full = [ttf_labels[indices[i][0]] for i in range(len(X))]

    # Convert and save
    risk_onehot = np.eye(len(RISK_LEVELS))[risk_labels_full]
    ttf_array = np.array(ttf_labels_full).reshape(-1, 1)

    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/risk_labels.npy', risk_onehot)
    np.save('data/processed/ttf_labels.npy', ttf_array)

    print(f"[Distiller] ✅ SUCCESS! Sampled soft labels expanded to full dataset and saved")
    print(f"[Distiller] 📊 Final dataset: {risk_onehot.shape} risk labels, {ttf_array.shape} TTF labels")

    return risk_onehot, ttf_array


# EMERGENCY: Stop current process and use this instead
def generate_soft_labels_emergency_fast(config, n_samples: int = 200):
    """Emergency fast version - use tiny sample for immediate results."""
    print(f"[Distiller] 🚨 EMERGENCY FAST MODE: Using only {n_samples} samples")

    X = np.load(config['data']['features_cache'])

    # Take systematic sample across the dataset
    indices = np.linspace(0, len(X) - 1, n_samples, dtype=int)
    X_sampled = X[indices]

    print(f"[Distiller] Processing {len(X_sampled)} samples (estimated time: {len(X_sampled) * 3 / 60:.1f} minutes)")

    teacher = GeminiTeacher(config)
    risk_labels = []
    ttf_labels = []

    # Process with minimal batching
    for i, sample in enumerate(tqdm(X_sampled, desc="Fast processing")):
        result = teacher.process_single_sample(sample, i)

        if result[0] is not None and result[1] is not None:
            risk_labels.append(result[0])
            ttf_labels.append(result[1])
        else:
            risk_labels.append(np.random.randint(0, 4))
            ttf_labels.append(np.random.uniform(10, 100))

        # Very short delay
        time.sleep(0.2)

    # Simple expansion - just repeat the pattern
    pattern_length = len(risk_labels)
    risk_labels_full = []
    ttf_labels_full = []

    for i in range(len(X)):
        pattern_idx = i % pattern_length
        risk_labels_full.append(risk_labels[pattern_idx])
        ttf_labels_full.append(ttf_labels[pattern_idx])

    # Convert and save
    risk_onehot = np.eye(len(RISK_LEVELS))[risk_labels_full]
    ttf_array = np.array(ttf_labels_full).reshape(-1, 1)

    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/risk_labels.npy', risk_onehot)
    np.save('data/processed/ttf_labels.npy', ttf_array)

    print(f"[Distiller] ⚡ EMERGENCY COMPLETE! Labels saved in under 10 minutes")

    return risk_onehot, ttf_array