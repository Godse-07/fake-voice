import joblib
from transformers import AutoFeatureExtractor, Wav2Vec2Model
import torch
import librosa
import numpy as np
import os
import torch.nn.functional as F
from scipy.special import expit
import tempfile
from fastapi import FastAPI, File, UploadFile
import uvicorn

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom model to truncate layers
class CustomWav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)
        self.encoder.layers = self.encoder.layers[:9]


# Load the truncated model
truncated_model = CustomWav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-2b")


# Define feature extractor using HuggingFace
class HuggingFaceFeatureExtractor:
    def __init__(self, model, feature_extractor_name):
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            feature_extractor_name
        )
        self.model = model
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, audio, sr):
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[9]


FEATURE_EXTRACTOR = HuggingFaceFeatureExtractor(
    truncated_model, "facebook/wav2vec2-xls-r-2b"
)

# Load classifier, scaler, and threshold from joblib file
classifier, scaler, thresh = joblib.load(
    "logreg_margin_pruning_ALL_with_scaler+threshold.joblib"
)


def segment_audio(audio, sr, segment_duration):
    segment_samples = int(segment_duration * sr)
    total_samples = len(audio)
    segments = [
        audio[i : i + segment_samples] for i in range(0, total_samples, segment_samples)
    ]
    return segments


def process_audio(input_data, segment_duration=10):
    audio, sr = librosa.load(input_data, sr=16000)
    if len(audio.shape) > 1:
        audio = audio[0]
    segments = segment_audio(audio, sr, segment_duration)
    segment_predictions = []
    output_lines = []
    eer_threshold = (
        thresh - 5e-3
    )  # small margin error due to feature extractor space differences
    for idx, segment in enumerate(segments):
        features = FEATURE_EXTRACTOR(segment, sr)
        features_avg = torch.mean(features, dim=1).cpu().numpy()
        features_avg = features_avg.reshape(1, -1)
        decision_score = classifier.decision_function(features_avg)
        decision_score_scaled = scaler.transform(
            decision_score.reshape(-1, 1)
        ).flatten()
        decision_value = decision_score_scaled[0]
        pred = 1 if decision_value >= eer_threshold else 0
        if pred == 1:
            confidence_percentage = expit(decision_score).item()
        else:
            confidence_percentage = 1 - expit(decision_score).item()
        segment_predictions.append(pred)
        line = f"Segment {idx + 1}: {'Real' if pred == 1 else 'Fake'} (Confidence: {np.round(confidence_percentage*100, 2)}%)"
        output_lines.append(line)
    overall_prediction = (
        1 if sum(segment_predictions) > (len(segment_predictions) / 2) else 0
    )
    overall_line = (
        f"Overall Prediction: {'Real' if overall_prediction == 1 else 'Fake'}"
    )
    output_str = overall_line + "\n" + "\n".join(output_lines)
    return output_str


# Create FastAPI app
app = FastAPI()


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    # Read the uploaded file and write to a temporary file
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Process the audio file and obtain predictions
    result = process_audio(tmp_path)

    # Clean up the temporary file
    os.remove(tmp_path)

    return {"prediction": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
