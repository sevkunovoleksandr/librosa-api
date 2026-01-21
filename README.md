# Librosa Async API

**Librosa Async API** is a FastAPI-based server for asynchronous audio analysis.  
It allows uploading audio files (MP3/WAV) and returns detailed audio features including tempo, beats, onsets, loudness, rhythmic events, and visualizations.

Optionally, if `madmom` is installed, the API can also detect downbeats for more precise rhythm analysis.

---

## Features

- Audio metadata extraction (title, artist, genre) using `mutagen`
- Beat and onset detection using `librosa`
- Tempo estimation and RMS (loudness) analysis
- Optional downbeat detection using `madmom`
- Harmonic / percussive source separation
- Base64-encoded PNG plots generation:
  - Onset strength with beats
  - PLP (Percussive-Like Peaks) with beats
  - Harmonic vs percussive components with downbeats
- JSON-based API responses suitable for automation and integrations

---

## Requirements

- Python 3.11 or higher
- Linux server (Ubuntu / Debian recommended)
- `ffmpeg` (required for MP3 decoding)
- Optional: `madmom` (downbeat detection)

---

## Installation

### System Dependencies

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip ffmpeg -y
```

### Clone and Setup

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Optional downbeat detection:

```bash
pip install madmom
```

---

## Configuration

The following values can be adjusted in `main.py`:

- `MAX_FILE_SIZE` — Maximum allowed upload size (default: 20 MB)
- `DEFAULT_SR` — Audio sampling rate (default: 22050 Hz)

Adjust these parameters to optimize memory usage and performance for large audio files.

---

## Running the Server

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

- API endpoint: `http://<server-ip>:8000`
- Interactive API documentation: `http://<server-ip>:8000/docs`

---

## API Usage

### Analyze Audio

**Endpoint:** `/analyze`  
**Method:** `POST`  
**Content-Type:** `multipart/form-data`

#### Parameters

- `audio` — MP3 or WAV file

#### cURL Example

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "audio=@/path/to/audio.mp3"
```

---

## Example JSON Response

```json
{
  "tempo": 120.5,
  "beat_times": [0.0, 0.5, 1.0],
  "onset_times": [0.1, 0.4, 0.9],
  "rms": [0.02, 0.03],
  "duration": 180.0,
  "downbeats": [0.0, 2.0],
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "events": [
    {
      "timeshow_id": 160,
      "event_id": "M1",
      "event_label": "measure",
      "time_stamp": 0.0,
      "event_color": "#F3F6EC",
      "Value": 1
    }
  ],
  "song_label": "Song Title",
  "artist": "Artist Name",
  "genre": "Genre"
}
```

---

## Deployment on Linux

### Run with Uvicorn

Development mode:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Production mode:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The `--workers 4` option allows handling multiple concurrent requests.

---

## Systemd Service (Auto-Start)

Create a systemd service file:

```bash
sudo nano /etc/systemd/system/librosa-api.service
```

```ini
[Unit]
Description=Librosa Async API
After=network.target

[Service]
User=youruser
WorkingDirectory=/path/to/repo
ExecStart=/path/to/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable librosa-api
sudo systemctl start librosa-api
sudo systemctl status librosa-api
```

---

## Notes

- MP3 support requires `ffmpeg`
- Downbeat detection is enabled only if `madmom` is installed
- Designed for backend automation, music analysis pipelines, and rhythm-based applications
