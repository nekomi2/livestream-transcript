import os
import subprocess
import threading
import time
import asyncio
import logging
import re
from datetime import datetime
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from yt_dlp import YoutubeDL
from pydub import AudioSegment
import dotenv
from openai import OpenAI

# Load environment variables from .env file
dotenv.load_dotenv()

# Configuration
API_KEY = dotenv.get_key(".env", "OPENAI_API_KEY")
YOUTUBE_URL = dotenv.get_key(".env", "YOUTUBE_URL")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")
if not YOUTUBE_URL:
    raise ValueError("YOUTUBE_URL not found in .env file.")

client = OpenAI(api_key=API_KEY)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Global variables to store the current transcript directory and stream title
CURRENT_TRANSCRIPT_DIR = None
STREAM_TITLE = "Live Transcription"

# Connection Manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)
        logging.info(f"WebSocket client connected: {websocket.client}")

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logging.info(f"WebSocket client disconnected: {websocket.client}")

    async def broadcast(self, message: str):
        async with self.lock:
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logging.error(f"Error sending message to {connection.client}: {e}")

manager = ConnectionManager()

def sanitize_filename(name):
    """
    Sanitize the stream title to create a valid filename.
    Removes or replaces characters that are invalid in file names.
    """
    sanitized = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", name)
    return sanitized.strip()[:255]

def get_direct_audio_url_and_title(youtube_url):
    """
    Uses yt_dlp to extract the direct audio stream URL and the livestream title.
    Returns a tuple of (direct_audio_url, stream_title).
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(youtube_url, download=False)
            stream_title = info.get("title", "live_transcript")
            # Iterate through formats to find audio-only
            audio_url = None
            for f in info["formats"]:
                if f.get("acodec") != "none" and f.get("vcodec") == "none":
                    logging.info(
                        f"Selected audio format: {f['format_id']} - {f['ext']} - {f.get('abr', 'unknown bitrate')} bitrate"
                    )
                    audio_url = f["url"]
                    break
            # Fallback to best audio if no audio-only formats are found
            if not audio_url:
                logging.warning(
                    "No audio-only formats found. Falling back to best available audio."
                )
                audio_url = info.get("url")
            return audio_url, stream_title
        except Exception as e:
            logging.error(f"Failed to extract audio URL and title: {e}")
            return None, None

def stream_audio(ffmpeg_input_url, output_file):
    """
    Streams audio from the direct audio stream URL to a raw PCM file using ffmpeg.
    """
    command = [
        "ffmpeg",
        "-i",
        ffmpeg_input_url,
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # Raw PCM audio
        "-ar",
        "16000",  # Sample rate (16kHz)
        "-ac",
        "1",  # Mono channel
        "-f",
        "s16le",  # Output format
        output_file,
        "-y",  # Overwrite output file if it exists
    ]
    logging.info("Starting audio stream with ffmpeg...")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg encountered an error: {e}")

def transcribe_audio(filename):
    """
    Sends the audio file to OpenAI's Whisper API for transcription.
    """
    logging.info(f"Transcribing audio segment: {filename}")
    try:
        with open(filename, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
        logging.info("Transcription completed.")
        return transcription.text
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        return None

def get_latest_transcript(transcript_file):
    """
    Retrieves the latest transcript content from the transcript file.
    """
    try:
        if not os.path.exists(transcript_file):
            logging.warning("Transcript file does not exist yet.")
            return ""
        with open(transcript_file, "r", encoding="utf-8") as f:
            content = f.read()
        logging.info(f"Retrieved latest transcript from {transcript_file}")
        return content
    except Exception as e:
        logging.error(f"Error retrieving latest transcript: {e}")
        return ""

def process_live_transcription(
    output_file, transcript_file, loop, fetch_duration=30, debug=False, keep_files=False
):
    """
    Monitors the raw audio file, reads new audio data based on the offset,
    and transcribes each new segment. Appends transcriptions to the transcript file
    and broadcasts updates via WebSocket.
    """
    try:
        # Ensure the transcript directory exists
        transcript_dir = os.path.dirname(transcript_file)
        os.makedirs(transcript_dir, exist_ok=True)

        # Wait until the output_file is created and has data
        while not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            logging.info(f"Waiting for {output_file} to be created and contain data...")
            time.sleep(1)

        with open(output_file, "rb") as f:
            # Define audio properties
            sample_rate = 16000  # Hz
            channels = 1  # Mono
            sample_width = 2  # bytes (16-bit)
            bytes_per_second = sample_rate * channels * sample_width
            bytes_per_fetch = bytes_per_second * fetch_duration

            logging.info(
                f"Audio properties: {sample_rate} Hz, {channels} channels, {sample_width * 8} bits"
            )
            logging.info(
                f"Bytes per second: {bytes_per_second}, Bytes per fetch: {bytes_per_fetch}"
            )

            last_read = 0  # Initialize offset

            while True:
                current_size = os.path.getsize(output_file)
                available_bytes = current_size - last_read

                if available_bytes >= bytes_per_fetch:
                    f.seek(last_read)
                    data = f.read(bytes_per_fetch)
                    last_read += bytes_per_fetch

                    # Convert raw PCM data to AudioSegment
                    audio_segment = AudioSegment(
                        data=data,
                        sample_width=sample_width,
                        frame_rate=sample_rate,
                        channels=channels,
                    )

                    # Export the segment to a temporary WAV file inside the transcript directory
                    temp_filename = os.path.join(
                        transcript_dir, f"temp_{int(time.time())}.wav"
                    )
                    audio_segment.export(temp_filename, format="wav")

                    if debug:
                        logging.info(
                            f"Debug mode: saved audio chunk to {temp_filename}"
                        )
                    else:
                        transcription = transcribe_audio(temp_filename)
                        if transcription:
                            # Append the transcription with a timestamp to the transcript file
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            with open(transcript_file, "a", encoding="utf-8") as tf:
                                tf.write(f"[{timestamp}] {transcription}\n")
                            logging.info(f"Transcription appended to {transcript_file}")

                            # Broadcast the transcription via WebSocket
                            asyncio.run_coroutine_threadsafe(
                                manager.broadcast(f"[{timestamp}] {transcription}"),
                                loop,
                            )
                        else:
                            logging.warning("No transcription available.")

                    if not keep_files:
                        try:
                            os.remove(temp_filename)
                            logging.info(f"Deleted temporary file {temp_filename}")
                        except OSError as e:
                            logging.error(
                                f"Error deleting file {temp_filename}: {e}"
                            )

                else:
                    # Not enough new data yet; wait briefly before checking again
                    time.sleep(1)

    except KeyboardInterrupt:
        logging.info("Stopped live transcription.")
    except Exception as e:
        logging.error(f"An error occurred in transcription process: {e}")

@app.on_event("startup")
async def startup_event():
    """
    Initialize the transcription process when the server starts.
    """
    global CURRENT_TRANSCRIPT_DIR  # Declare as global to modify the global variable
    global STREAM_TITLE  # Declare as global to modify the global variable

    direct_audio_url, stream_title = get_direct_audio_url_and_title(YOUTUBE_URL)
    if not direct_audio_url:
        logging.error("Could not retrieve direct audio stream URL. Exiting startup.")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve audio stream URL."
        )

    logging.info(f"Livestream Title: {stream_title}")
    STREAM_TITLE = stream_title  # Update the global variable

    # Sanitize the stream title to create a valid directory name
    sanitized_title = sanitize_filename(stream_title)

    # Create a timestamp for the directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # New naming convention: transcripts_{YYYYMMDD_HHMMSS}_{sanitized_title}
    transcript_dir = f"transcripts_{timestamp}_{sanitized_title}"
    os.makedirs(transcript_dir, exist_ok=True)

    logging.info(f"Transcripts will be saved to directory: {transcript_dir}")

    # Assign the current transcription directory to the global variable
    CURRENT_TRANSCRIPT_DIR = transcript_dir

    # Define the path for the raw audio file and the single transcript file inside the transcript directory
    output_file = os.path.join(transcript_dir, "live_audio.raw")
    transcript_file = os.path.join(transcript_dir, "transcript.txt")

    # Get the current running event loop
    loop = asyncio.get_running_loop()

    # Start ffmpeg in a separate thread
    ffmpeg_thread = threading.Thread(
        target=stream_audio, args=(direct_audio_url, output_file), daemon=True
    )
    ffmpeg_thread.start()

    # Start processing transcription in a separate thread
    transcription_thread = threading.Thread(
        target=process_live_transcription,
        args=(output_file, transcript_file, loop),
        daemon=True,
    )
    transcription_thread.start()

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    """
    Serve the HTML page for live transcription.
    """
    global STREAM_TITLE

    return templates.TemplateResponse("index.html", {
        "request": request,
        "stream_title": STREAM_TITLE
    })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to send live transcription updates to clients.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection open
            data = await websocket.receive_text()
            # Optionally handle messages from the client
            logging.info(f"Received message from client: {data}")
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket connection error: {e}")
        await manager.disconnect(websocket)

@app.get("/latest_transcript", response_class=JSONResponse)
async def latest_transcript():
    """
    API endpoint to retrieve the entire transcript content.
    """
    global CURRENT_TRANSCRIPT_DIR

    if not CURRENT_TRANSCRIPT_DIR:
        return JSONResponse(content={"error": "Transcript directory not set."}, status_code=500)

    transcript_file = os.path.join(CURRENT_TRANSCRIPT_DIR, "transcript.txt")

    content = get_latest_transcript(transcript_file)
    if content == "":
        return JSONResponse(content={"error": "No transcripts found."}, status_code=404)

    return JSONResponse(content={"content": content})
