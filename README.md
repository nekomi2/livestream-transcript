# Livestream transcript
To help with prop betting on speeches, I'm making a small tool (with the help of ChatGippity) to transcribe livestreams in real time and host it on the web. I'll add better admin tools for changing streams, seeing past transcripts, and word counts.
## Installation
Create venv
```
python -m venv virtual
source virtual/bin/activate
pip install -r requirements.txt
```

Run server
```
uvicorn server:app --host 0.0.0.0 --port 8000
```


