# Create a hosted microservice using wav2vec2-large-960h to deploy an Automatic Speech Recognition (ASR) AI model that can be used to transcribe audio files
# Please ensure that your speech input is also sampled at 16kHz.
# Documentation for FastAPI: https://fastapi.tiangolo
# Documentation for wav2vec2: https://huggingface.co/facebook/wav2vec2-large-960h

# 2b Write a ping API (i.e. http://localhost:8001/ping via GET) to return a response of “pong” to check if your service is working.
# 2c Write an API with the following specifications as a hosted inference API for the model in Task 2a. Name your file asr_api.py.

from fastapi import FastAPI, File, UploadFile
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from io import BytesIO
import os

app = FastAPI()

# Define the device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the model and processor, and move the model to the selected device
model_name = "/home/shawnyzy/Documents/GitHub/shawn-ds-app-sub/asr-train/wav2vec2-large-960h-cv"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

# Ping API to check if the service is working
@app.get("/ping")
def ping():
    return {"response": "pong"}

# Single-file ASR API
@app.post("/asr")
async def asr(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_location = f"/tmp/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Convert the audio to 16kHz mono (check if this is needed!!!!)
        # CHECK IF THIS IS NEEDED
        audio = AudioSegment.from_file(file_location, format="mp3").set_frame_rate(16000).set_channels(1)
        samples = torch.tensor(audio.get_array_of_samples()).float() / (2**15)

        # Prepare inputs and move to the device
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt", padding=True).to(device)

        # Perform inference
        with torch.no_grad():
            logits = model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])

        # Calculate audio duration
        duration = len(audio) / 1000.0

        # Delete the temporary file
        os.remove(file_location)

        return {
            "transcription": transcription,
            "duration": f"{duration:.2f}"
        }
    except Exception as e:
        return {"error": str(e)}

# Batch ASR API
@app.post("/asr_batch")
async def asr_batch(files: list[UploadFile] = File(...)):
    transcriptions = []
    try:
        for file in files:
            # Save each uploaded file temporarily
            file_location = f"/tmp/{file.filename}"
            with open(file_location, "wb") as f:
                f.write(await file.read())

            # Convert the audio to 16kHz mono
            audio = AudioSegment.from_file(file_location, format="mp3").set_frame_rate(16000).set_channels(1)
            samples = torch.tensor(audio.get_array_of_samples()).float() / (2**15)

            # Prepare inputs and move to the device
            inputs = processor(samples, sampling_rate=16000, return_tensors="pt", padding=True).to(device)

            # Perform inference
            with torch.no_grad():
                logits = model(inputs.input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.decode(predicted_ids[0])

            # Append transcription results
            transcriptions.append({
                "file": file.filename,
                "transcription": transcription,
                "duration": len(audio) / 1000.0
            })

            # Delete the temporary file
            os.remove(file_location)

        return {"results": transcriptions}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
