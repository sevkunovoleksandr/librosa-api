from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import gc
import base64
from mutagen import File as MutagenFile
import soundfile as sf
import tempfile
import os
import logging

# ---- madmom imports (protected)
try:
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    madmom_available = True
except Exception as e:
    madmom_available = False
    logging.warning(f"madmom import failed: {e}")

# ---- Logging ----
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

app = FastAPI(title="Librosa Async API")

# ---- Настройки ----
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
DEFAULT_SR = 22050  # зменшення sampling rate для економії пам'яті

# ---- Utilities ----
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    else:
        return obj

def get_metadata(file_stream):
    try:
        file_stream.seek(0)
        audio = MutagenFile(file_stream)
        if audio is None or not audio.tags:
            return {'song_label': None, 'artist': None, 'genre': None}
        tags = audio.tags
        def get_tag_value(tag_name):
            if tag_name in tags:
                val = tags[tag_name]
                if isinstance(val, (list, tuple)):
                    return str(val[0])
                else:
                    return str(val)
            return None
        return {
            'song_label': get_tag_value('TIT2'),
            'artist': get_tag_value('TPE1'),
            'genre': get_tag_value('TCON'),
        }
    except Exception as e:
        logging.warning(f"Metadata extraction failed: {e}")
        return {'song_label': None, 'artist': None, 'genre': None}

# ---- Main route ----
@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):

    content = await audio.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    tmp_path = None
    img_base64 = None
    file_like = io.BytesIO(content)

    try:
        # ---- Metadata ----
        metadata = get_metadata(file_like)

        # ---- Load audio (mono + lower sr) ----
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_mp3:
            tmp_mp3.write(content)
            tmp_mp3.flush()
            tmp_path_for_load = tmp_mp3.name

        y, sr = librosa.load(tmp_path_for_load, sr=DEFAULT_SR, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        hop_length = 512

        # ---- Beat/onset analysis ----
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]

        # ---- PLP ----
        plp = librosa.beat.plp(y=y, sr=sr, hop_length=hop_length)
        plp_times = librosa.times_like(plp, sr=sr, hop_length=hop_length)
        plp_peak_mask = librosa.util.localmax(plp)
        plp_beats = np.flatnonzero(plp_peak_mask)
        plp_beat_times = librosa.frames_to_time(plp_beats, sr=sr, hop_length=hop_length)

        # ---- Madmom Downbeats ----
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
            tmp_path = tmp_wav.name
            # якщо моно, дублюємо канал
            if y.ndim == 1:
                y_stereo = np.vstack([y, y])
            else:
                y_stereo = y
            sf.write(tmp_path, y_stereo.T, sr, subtype='PCM_16')

        downbeats = []
        if madmom_available:
            try:
                proc = RNNDownBeatProcessor()
                act = proc(tmp_path)
                beats_madmom = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)(act)
                logging.info(f"beats_madmom raw: {beats_madmom}")
                downbeats = [float(t) for t, b in beats_madmom if b == 1]
                logging.info(f"downbeats (s): {downbeats}")
                downbeats = [db for db in downbeats if db <= duration]
                if not downbeats:
                    logging.warning("Madmom не знайшов жодного downbeat або вони поза duration")
            except Exception as e:
                logging.warning(f"madmom failed: {e}")
        else:
            logging.warning("madmom not available")

        # ---- Plot graphs ----
        fig = None
        img_buf = io.BytesIO()
        try:
            dpi = 100
            max_pixels = 15000
            max_inches = max_pixels / dpi
            base_width = 10
            extra_width = duration / 1
            fig_width = min(base_width + extra_width, max_inches)

            fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(fig_width, 8))

            ax[0].plot(times, onset_env, label='Onset strength')
            if beat_times.size:
                ax[0].vlines(beat_times, 0, onset_env.max(), color='r', linestyle='--', label='librosa Beats', linewidth=1, alpha=0.7)
            if downbeats:
                ax[0].vlines(downbeats, 0, onset_env.max(), color='g', linestyle='-', label='madmom Downbeats', linewidth=2, alpha=0.8)
            ax[0].legend(loc='upper left')
            ax[0].set_title('Onset strength + Beats (librosa vs madmom)', loc='left')

            ax[1].plot(plp_times, plp, label='PLP')
            if plp_beat_times.size:
                ax[1].vlines(plp_beat_times, 0, plp.max(), color='r', linestyle='--', label='PLP Beats', linewidth=1, alpha=0.7)
            if downbeats:
                ax[1].vlines(downbeats, 0, plp.max(), color='g', linestyle='-', label='madmom Downbeats', linewidth=2, alpha=0.8)
            ax[1].legend(loc='upper left')
            ax[1].set_title('PLP (librosa) + madmom Downbeats', loc='left')

            y_harm, y_perc = librosa.effects.hpss(y)
            times_harm = np.linspace(0, duration, len(y))
            ax[2].plot(times_harm, y_harm, color='b', alpha=0.6, label='Harmonic')
            ax[2].plot(times_harm, y_perc, color='r', alpha=0.6, label='Percussive')
            if downbeats:
                ax[2].vlines(downbeats, -0.3, 0.3, color='g', linestyle='-', label='madmom Downbeats', linewidth=2, alpha=0.8)
            ax[2].legend(loc='upper left')
            ax[2].set_title('Harmonic (blue) vs Percussive (red) + Downbeats', loc='left')

            ax[0].set_xlim(0, duration)
            plt.tight_layout()
            plt.savefig(img_buf, format='png', dpi=dpi)
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
            logging.info(f"Plotted {len(downbeats)} Madmom downbeats on the graphs")
        finally:
            if fig:
                plt.close(fig)
            if img_buf:
                del img_buf
            gc.collect()

        # ---- Events ----
        events = []
        label_counts = {}
        for idx, t in enumerate(beat_times, start=1):
            action = "Go+" if any(abs(t - db) < 0.05 for db in downbeats) else "fillIn"
            # event_label = f"{action}_{label_counts.get(action,0)}" if action in label_counts else action
            event_label = "measure"
            label_counts[action] = label_counts.get(action,0)+1
            events.append({
                "timeshow_id": 160,
                "event_id": f"M{idx}",
                "event_label": event_label,
                "time_stamp": round(float(t), 3),
                "event_color": "#F3F6EC",
                "Value": 1
            })

        # ---- Response ----
        response_data = {
            'tempo': float(tempo),
            'beat_times': convert_ndarray(beat_times),
            'onset_times': convert_ndarray(onset_times),
            'rms': convert_ndarray(rms),
            'duration': float(duration),
            'downbeats': downbeats,
            'image_base64': img_base64,
            'speed': float(tempo),
            'length': float(duration),
            'events': events,
            **metadata
        }

        return JSONResponse(response_data)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logging.warning(f"Could not delete temp file: {tmp_path} ({e})")

        if 'tmp_path_for_load' in locals() and os.path.exists(tmp_path_for_load):
            try:
                os.remove(tmp_path_for_load)
            except Exception as e:
                logging.warning(f"Could not delete temp file: {tmp_path_for_load} ({e})")

        try:
            del y, y_stereo, onset_env, plp, beat_times, onset_times
        except NameError:
            pass
        gc.collect()