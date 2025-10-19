# 🎤 Steno

A CLI tool for real-time transcription of microphone and system audio with automatic voice identification.

## Features

- **Dual Audio Sources**: Simultaneously capture microphone input and system audio (speakers/loopback)
- **Real-Time Transcription**: Fast transcription using Whisper with <1s latency
- **Voice Identification**: Automatic voice labeling using ECAPA-TDNN voice embeddings
- **Voice Activity Detection**: Intelligent VAD using Silero to filter silence
- **JSONL Output**: Timestamped transcriptions with voice IDs in JSON Lines format
- **Verbose Mode**: Real-time audio level meters for monitoring input
- **Auto-Detection**: Automatically finds appropriate audio devices

## Requirements

- Python 3.11+
- Windows (for system audio loopback via WASAPI)
- ~500MB disk space for models (downloaded automatically on first run)

## Installation

### Using `uv` (Recommended)

Install globally using the `uv` tool:
```bash
uv tool install --editable . --with torchaudio --with speechbrain
```

This will make the `steno` command available globally on your system.

2. Download models (happens automatically on first run):
   - Whisper tiny model (~75MB)
   - ECAPA-TDNN voice embeddings (~20MB)
   - Silero VAD (~2MB)

## Usage

### Basic Examples

**Transcribe microphone only:**
```bash
steno -m -o transcript.jsonl
```

**Transcribe system audio (speakers) only:**
```bash
steno -s -o transcript.jsonl
```

**Transcribe both microphone and speakers:**
```bash
steno -m -s -o transcript.jsonl
```

**Verbose mode with audio meters:**
```bash
steno -m -s -o transcript.jsonl -v
```

*Note: If you're using local development mode (not globally installed), use `python steno.py` instead of `steno`.*

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-m, --mic [DEVICE_ID]` | Enable microphone capture (optional device ID) | Auto-detect |
| `-s, --speaker [DEVICE_ID]` | Enable system audio capture (optional device ID) | Auto-detect |
| `-o, --out FILE` | Output JSONL file path | **Required** |
| `-t, --threshold VALUE` | Voice similarity threshold (0.3-0.8) | 0.36 |
| `-v, --verbose` | Show real-time audio meters and transcriptions | Off |
| `-h, --help` | Show help message | - |

### Device Selection

**Auto-detection (recommended):**
```bash
steno -m -s -o output.jsonl
```

**Specify device IDs:**
```bash
steno -m 1 -s 5 -o output.jsonl
```

To find available device IDs, run:
```python
import sounddevice as sd
print(sd.query_devices())
```

### Voice Identification Tuning

The `-t/--threshold` parameter controls voice similarity (cosine distance):

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| `0.3` | Merge similar voices | Few voices, reduce false splits |
| `0.36` (default) | Balanced grouping | General purpose |
| `0.5` | Distinguish voices | Many voices, prevent merging |
| `0.7-0.8` | Very strict | Maximum voice separation |

**Example:**
```bash
# More aggressive voice grouping (fewer voice IDs)
steno -m -s -o output.jsonl -t 0.3

# Stricter voice separation (more voice IDs)
steno -m -s -o output.jsonl -t 0.6
```

## Output Format

Transcriptions are saved in **JSON Lines** format with one JSON object per line:

```json
{"timestamp": "2024-01-15T10:30:45.123456", "source": "mic", "speaker": "a1b2c3", "text": "Hello, how are you?"}
{"timestamp": "2024-01-15T10:30:47.654321", "source": "speaker", "speaker": "d4e5f6", "text": "I'm doing well, thanks."}
```

### Fields

- `timestamp`: ISO 8601 timestamp with microsecond precision
- `source`: Audio source (`"mic"` or `"speaker"`)
- `voice`: 6-character voice hash ID (consistent within session)
- `text`: Transcribed text

## Verbose Mode

When running with `-v/--verbose`, you'll see:

```
[MIC    ] ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  24%
[SPEAKER] ███████████████████████████████████████░░░░░░░░░  78%
[2024-01-15 10:30:45] [speaker] [d4e5f6] I'm doing well, thanks.
```

- Real-time audio level meters (updated 20 times per second)
- Transcriptions logged to stdout as they complete
- Meters stay at the bottom, transcriptions scroll above

## Project Structure

```
src3/
├── steno.py              # Main CLI tool (~500 lines)
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── lib/                 # Voice identification modules
    ├── embeddings.py    # ECAPA-TDNN wrapper
    └── identification.py # Voice tracking logic
```

## Technical Details

### Audio Processing Pipeline

1. **Capture**: Simultaneous mic (sounddevice) + speakers (pyaudiowpatch/WASAPI)
2. **VAD**: Silero filters silence, accumulates speech chunks
3. **Transcription**: Whisper tiny model (int8 quantized, CPU-only)
4. **Voice ID**: ECAPA-TDNN extracts 192-dim embeddings
5. **Clustering**: Cosine similarity groups voices (threshold-based)
6. **Output**: JSONL append with timestamp, source, voice, text

### Performance

- **Latency**: <1s from speech end to transcription
- **CPU Usage**: ~15-30% on modern CPUs (Whisper tiny)
- **Memory**: ~2GB RAM (models + buffers)
- **Throughput**: Real-time (1x playback speed)

### Models

| Model | Size | Purpose | Source |
|-------|------|---------|--------|
| Whisper Tiny | ~75MB | Speech-to-text | faster-whisper |
| ECAPA-TDNN | ~20MB | Voice embeddings | speechbrain |
| Silero VAD | ~2MB | Voice activity detection | silero-vad |

## Limitations

- **Windows only** for system audio capture (pyaudiowpatch uses WASAPI)
- **Single session IDs**: Voice hashes reset between runs
- **CPU inference**: No GPU acceleration (for portability)
- **English optimized**: Whisper tiny trained primarily on English

## Troubleshooting

**No microphone detected:**
- Check `python -c "import sounddevice; print(sounddevice.query_devices())"`
- Specify device ID manually: `-m 0`

**No system audio detected:**
- Ensure audio is playing during detection
- Check loopback device: `python -c "import pyaudiowpatch as pa; print(pa.PyAudio().get_default_wasapi_loopback())"`
- Specify device ID manually: `-s 5`

**Voice IDs change too often:**
- Lower threshold: `-t 0.3` (groups similar voices)

**Multiple voice merged into one ID:**
- Raise threshold: `-t 0.6` (splits voices more aggressively)

**"Input audio chunk is too short" errors:**
- This is normal during silence/pauses - VAD accumulates audio until sufficient

## License

See project root LICENSE file.

## Credits

Built with:
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Whisper inference
- [speechbrain](https://speechbrain.github.io/) - Voice embeddings
- [silero-vad](https://github.com/snakers4/silero-vad) - Voice activity detection
- [pyaudiowpatch](https://github.com/s0d3s/PyAudioWPatch) - WASAPI loopback

  - `identification.py`: Voice identification and tracking

## Requirements

- Python 3.10+
- Windows (for WASAPI loopback support on system audio)
- Microphone and/or speakers/headphones

## Notes

- System audio capture requires WASAPI loopback support (Windows)
- First run will download the Whisper and ECAPA-TDNN models (~100MB total)
- Use Ctrl+C to stop transcription gracefully
