from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import whisper
from gtts import gTTS
import tempfile
import uuid

app = FastAPI()

# Load Whisper model (lightweight)
model = whisper.load_model("base")  # base is small enough for free Render

# -------------------------
# SPEECH TO TEXT
# -------------------------
@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        result = model.transcribe(tmp.name)

    return {"text": result["text"]}

# -------------------------
# TEXT TO SPEECH
# -------------------------
@app.post("/tts")
async def text_to_speech(text: str):
    filename = f"{uuid.uuid4()}.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return FileResponse(filename, media_type="audio/mpeg")
