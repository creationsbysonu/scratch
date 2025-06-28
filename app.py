from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import librosa
import logging
import traceback
import os
import tempfile

app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model = tf.keras.models.load_model("final_stutter_detection_cnn.h5")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# Constants
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.5
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
TARGET_SHAPE = (128, 128)

# Extract log-mel spectrogram
def extract_log_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=TARGET_SHAPE[0])
    log_mel = librosa.power_to_db(mel)
    padded = np.zeros(TARGET_SHAPE)
    h, w = log_mel.shape
    padded[:min(h, TARGET_SHAPE[0]), :min(w, TARGET_SHAPE[1])] = log_mel[:TARGET_SHAPE[0], :TARGET_SHAPE[1]]
    return padded

@app.get("/")
async def root():
    return {"message": "Stutter Detection API running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...), chunk_number: int = Form(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        audio_data = await file.read()

        if len(audio_data) < 100:
            raise HTTPException(status_code=400, detail="Uploaded file is empty or too small")

        # Save audio to temporary WAV file directly (assumes frontend sends .wav)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav.write(audio_data)
            temp_wav.flush()
            wav_path = temp_wav.name

        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

        os.remove(wav_path)

        if len(y) < CHUNK_SIZE:
            y = np.pad(y, (0, CHUNK_SIZE - len(y)))
        else:
            y = y[:CHUNK_SIZE]

        log_mel = extract_log_mel(y, sr)
        log_mel = log_mel[np.newaxis, ..., np.newaxis]

        pred = model.predict(log_mel, verbose=0)
        prob = float(pred[0][0])
        stutter = prob > 0.5

        return {
            "chunk_number": chunk_number,
            "probability": prob,
            "stutter_detected": stutter
        }

    except Exception as e:
        logging.error("Prediction error:", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
