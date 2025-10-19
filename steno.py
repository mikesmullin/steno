#!/usr/bin/env python3
"""
Steno - Real-time Audio Transcription with Speaker Identification

CLI tool for transcribing microphone and/or system audio with voice identification.
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import sounddevice as sd  # For microphone
import pyaudiowpatch as pyaudio  # For system audio (WASAPI loopback)
from faster_whisper import WhisperModel

# Import speaker identification modules
from lib.embeddings import SpeakerEmbeddings
from lib.identification import SpeakerIdentifier


class AudioTranscriber:
    """Real-time audio transcriber with speaker identification"""
    
    def __init__(
        self,
        mic_device: int = None,
        speaker_device: int = None,
        output_file: Path = None,
        speaker_threshold: float = 0.36,
        verbose: bool = False
    ):
        """
        Initialize transcriber
        
        Args:
            mic_device: Microphone device index (None to skip)
            speaker_device: Speaker loopback device index (None to skip)
            output_file: Output JSONL file path
            speaker_threshold: Similarity threshold for speaker matching
            verbose: Print transcriptions and audio meters to stdout
        """
        self.mic_device = mic_device
        self.speaker_device = speaker_device
        self.output_file = output_file or Path.cwd() / "transcript.jsonl"
        self.speaker_threshold = speaker_threshold
        self.verbose = verbose
        
        self.is_running = False
        self.sample_rate = 16000
        
        # Buffers for audio processing
        self.mic_buffer = []
        self.mic_vad_buffer = []
        self.mic_is_speaking = False
        self.mic_silence_count = 0
        
        self.system_buffer = []
        self.system_vad_buffer = []
        self.system_is_speaking = False
        self.system_silence_count = 0
        
        self.max_silence_chunks = 15
        self.vad_chunk_size = 512
        
        # Statistics
        self.transcription_count = 0
        
        # Audio level tracking for verbose mode
        self.mic_level = 0.0
        self.speaker_level = 0.0
        self.level_lock = threading.Lock()
        
        # System audio device info (will create PyAudio instance in thread)
        self.system_native_rate = None
        self.system_native_channels = None
        if speaker_device is not None:
            p_temp = pyaudio.PyAudio()
            device_info = p_temp.get_device_info_by_index(speaker_device)
            self.system_native_rate = int(device_info['defaultSampleRate'])
            self.system_native_channels = device_info['maxInputChannels']
            p_temp.terminate()
        
        # Load models
        self._load_models()
        
        # Create output file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'a', encoding='utf-8') as f:
            entry = {
                "speaker_id": "system",
                "text": f"[Steno transcription started]",
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "confidence": 1.0,
                "duration": 0.0
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"✓ Output: {self.output_file.absolute()}")
    
    def _load_models(self):
        """Load VAD, Whisper, and speaker identification models"""
        print("Loading models...")
        
        # Load Silero VAD
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            verbose=False
        )
        print("✓ VAD loaded")
        
        # Load Whisper
        self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✓ Whisper loaded")
        
        # Load speaker identification
        try:
            self.speaker_embeddings = SpeakerEmbeddings(device="cpu")
            print("✓ ECAPA-TDNN speaker embeddings loaded")
        except Exception as e:
            print(f"Warning: Failed to load ECAPA model: {e}")
            print("Speaker identification will not be available")
            self.speaker_embeddings = None
        
        self.speaker_identifier = SpeakerIdentifier(
            similarity_threshold=self.speaker_threshold,
            ttl_hours=1.0
        )
        print(f"✓ Speaker identifier ready (threshold={self.speaker_threshold})")
    
    def resample_audio(self, audio_data, orig_rate, target_rate):
        """Simple resampling"""
        if orig_rate == target_rate:
            return audio_data
        
        duration = len(audio_data) / orig_rate
        target_length = int(duration * target_rate)
        indices = np.linspace(0, len(audio_data) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)
        return resampled.astype(np.float32)
    
    def process_mic_audio(self, audio_chunk):
        """Process microphone audio with VAD"""
        try:
            # Update audio level for verbose mode
            if self.verbose:
                rms = np.sqrt(np.mean(audio_chunk ** 2))
                with self.level_lock:
                    self.mic_level = rms
            
            audio_tensor = torch.from_numpy(audio_chunk).float()
            with torch.no_grad():
                speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
            
            is_speech = speech_prob > 0.5
            
            if is_speech:
                if not self.mic_is_speaking:
                    logging.debug(f"🎤 MIC: Speech detected ({speech_prob:.3f})")
                    self.mic_is_speaking = True
                
                self.mic_buffer.append(audio_chunk)
                self.mic_silence_count = 0
            else:
                if self.mic_is_speaking:
                    self.mic_silence_count += 1
                    self.mic_buffer.append(audio_chunk)
                    
                    if self.mic_silence_count >= self.max_silence_chunks:
                        complete_audio = np.concatenate(self.mic_buffer)
                        duration = len(complete_audio) / self.sample_rate
                        
                        # Skip very short utterances
                        if duration >= 1.5:
                            logging.debug(f"📝 MIC: Speech ended ({duration:.1f}s)")
                            threading.Thread(
                                target=self.transcribe_and_identify,
                                args=(complete_audio, "microphone"),
                                daemon=True
                            ).start()
                        
                        self.mic_buffer = []
                        self.mic_is_speaking = False
                        self.mic_silence_count = 0
        
        except Exception as e:
            logging.error(f"Error processing mic audio: {e}")
    
    def process_system_audio(self, audio_chunk):
        """Process system audio with VAD"""
        try:
            # Resample if needed
            if self.system_native_rate != self.sample_rate:
                audio_chunk = self.resample_audio(audio_chunk, self.system_native_rate, self.sample_rate)
            
            # Update audio level for verbose mode
            if self.verbose:
                rms = np.sqrt(np.mean(audio_chunk ** 2))
                with self.level_lock:
                    self.speaker_level = rms
            
            # Accumulate chunks for VAD
            self.system_vad_buffer.extend(audio_chunk)
            
            # Process in 512-sample chunks
            while len(self.system_vad_buffer) >= self.vad_chunk_size:
                vad_chunk = np.array(self.system_vad_buffer[:self.vad_chunk_size])
                self.system_vad_buffer = self.system_vad_buffer[self.vad_chunk_size:]
                
                audio_tensor = torch.from_numpy(vad_chunk).float()
                with torch.no_grad():
                    speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
                
                is_speech = speech_prob > 0.5
                
                if is_speech:
                    if not self.system_is_speaking:
                        logging.debug(f"🔊 SYSTEM: Speech detected ({speech_prob:.3f})")
                        self.system_is_speaking = True
                    
                    self.system_buffer.append(vad_chunk)
                    self.system_silence_count = 0
                else:
                    if self.system_is_speaking:
                        self.system_silence_count += 1
                        self.system_buffer.append(vad_chunk)
                        
                        if self.system_silence_count >= self.max_silence_chunks:
                            complete_audio = np.concatenate(self.system_buffer)
                            duration = len(complete_audio) / self.sample_rate
                            
                            # Skip very short utterances
                            if duration >= 1.5:
                                logging.debug(f"📝 SYSTEM: Speech ended ({duration:.1f}s)")
                                threading.Thread(
                                    target=self.transcribe_and_identify,
                                    args=(complete_audio, "system_audio"),
                                    daemon=True
                                ).start()
                            
                            self.system_buffer = []
                            self.system_is_speaking = False
                            self.system_silence_count = 0
        
        except Exception as e:
            logging.error(f"Error processing system audio: {e}")
    
    def transcribe_and_identify(self, audio, source):
        """Transcribe and identify speaker"""
        try:
            # Transcribe
            segments, info = self.whisper_model.transcribe(
                audio,
                language="en",
                beam_size=5
            )
            
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            full_text = " ".join(text_parts)
            
            if not full_text:
                return
            
            # Extract speaker embedding
            speaker_id = "unknown"
            if self.speaker_embeddings:
                embedding = self.speaker_embeddings.extract(audio, self.sample_rate)
                
                if embedding is not None:
                    speaker_id = self.speaker_identifier.identify(embedding)
            
            # Save to file
            entry = {
                "speaker_id": speaker_id,
                "text": full_text,
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "confidence": 0.9,
                "duration": len(audio) / self.sample_rate,
                "source": source
            }
            
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            self.transcription_count += 1
            
            # Print to stdout if verbose
            if self.verbose:
                # Clear the meter line, print transcription, then redraw meter
                self._clear_meter_line()
                print(f"[{speaker_id}] ({source}): {full_text}")
                sys.stdout.flush()
        
        except Exception as e:
            logging.error(f"Error in transcription: {e}")
    
    def start(self):
        """Start transcription"""
        self.is_running = True
        
        threads = []
        
        # Start audio meter display if verbose
        if self.verbose:
            meter_thread = threading.Thread(target=self._display_audio_meters, daemon=True)
            meter_thread.start()
            threads.append(meter_thread)
        
        # Start microphone stream
        if self.mic_device is not None:
            thread = threading.Thread(target=self._run_mic_stream, daemon=True)
            thread.start()
            threads.append(thread)
            print(f"✓ Microphone stream started (device #{self.mic_device})")
        
        # Start system audio stream
        if self.speaker_device is not None:
            thread = threading.Thread(target=self._run_system_stream, daemon=True)
            thread.start()
            threads.append(thread)
            print(f"✓ System audio stream started (device #{self.speaker_device})")
        
        if self.verbose:
            print("\n🎙️  Listening with audio meters... (Ctrl+C to stop)\n")
        else:
            print("\n🎙️  Listening... (Ctrl+C to stop)\n")
        
        # Keep running
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")
            self.stop()
    
    def _clear_meter_line(self):
        """Clear the current meter line"""
        # Print carriage return and spaces to clear the line
        print("\r" + " " * 150, end="\r", flush=True)
    
    def _display_audio_meters(self):
        """Display real-time audio level meters (verbose mode only)"""
        while self.is_running:
            with self.level_lock:
                mic_level = self.mic_level
                speaker_level = self.speaker_level
            
            # Create visual meters (50 characters each)
            mic_meter_length = int(mic_level * 1000)
            speaker_meter_length = int(speaker_level * 1000)
            
            mic_meter = "█" * min(mic_meter_length, 50)
            speaker_meter = "█" * min(speaker_meter_length, 50)
            
            # Build display lines
            lines = []
            if self.mic_device is not None:
                lines.append(f"🎤 MIC:     {mic_meter:<50} {mic_level:.4f}")
            if self.speaker_device is not None:
                lines.append(f"🔊 SPEAKER: {speaker_meter:<50} {speaker_level:.4f}")
            
            # Print with carriage return to overwrite
            if lines:
                output = "\r" + " | ".join(lines)
                print(output, end="", flush=True)
            
            time.sleep(0.05)  # Update 20 times per second
    
    def _run_mic_stream(self):
        """Run microphone stream"""
        def callback(indata, frames, time_info, status):
            if status:
                logging.warning(f"Mic status: {status}")
            
            audio_chunk = indata[:, 0].copy()
            self.process_mic_audio(audio_chunk)
        
        with sd.InputStream(
            device=self.mic_device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=512,
            callback=callback
        ):
            while self.is_running:
                time.sleep(0.1)
    
    def _run_system_stream(self):
        """Run system audio stream"""
        # Create PyAudio instance in this thread
        p = pyaudio.PyAudio()
        
        def callback(in_data, frame_count, time_info, status):
            if status:
                logging.warning(f"System status: {status}")
            
            audio_chunk = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Convert stereo to mono if needed
            if self.system_native_channels == 2:
                audio_chunk = audio_chunk.reshape(-1, 2).mean(axis=1)
            
            self.process_system_audio(audio_chunk)
            
            return (in_data, pyaudio.paContinue)
        
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.system_native_channels,
            rate=self.system_native_rate,
            frames_per_buffer=512,
            input=True,
            input_device_index=self.speaker_device,
            stream_callback=callback
        )
        
        while self.is_running:
            time.sleep(0.1)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def stop(self):
        """Stop transcription"""
        self.is_running = False
        
        print(f"\n✓ Transcription completed")
        print(f"✓ Total transcriptions: {self.transcription_count}")
        print(f"✓ Output file: {self.output_file}")


def auto_detect_microphone():
    """Auto-detect default microphone"""
    try:
        default_idx = sd.default.device[0]
        device_info = sd.query_devices(default_idx)
        if device_info['max_input_channels'] > 0:
            print(f"✓ Auto-detected microphone: [{default_idx}] {device_info['name']}")
            return default_idx
    except:
        pass
    
    # Fallback: find first microphone
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0 and (
            'microphone' in device['name'].lower() or 
            'headset' in device['name'].lower() or
            'mic' in device['name'].lower()
        ):
            print(f"✓ Auto-detected microphone: [{idx}] {device['name']}")
            return idx
    
    return None


def auto_detect_speaker():
    """Auto-detect default speaker loopback device"""
    try:
        p = pyaudio.PyAudio()
        
        # Try to get default loopback
        try:
            default_loopback = p.get_default_wasapi_loopback()
            idx = default_loopback['index']
            device_info = p.get_device_info_by_index(idx)
            print(f"✓ Auto-detected speaker: [{idx}] {device_info['name']}")
            return idx, p
        except:
            pass
        
        # Fallback: find first loopback device
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            host_api = p.get_host_api_info_by_index(info['hostApi'])
            
            if 'WASAPI' in host_api['name'] and info.get('isLoopbackDevice', False):
                print(f"✓ Auto-detected speaker: [{i}] {info['name']}")
                return i, p
        
        return None, p
    except:
        return None, None


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Steno - Real-time audio transcription with speaker identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe from microphone (auto-detect)
  python steno.py --mic
  
  # Transcribe from system audio (auto-detect)
  python steno.py --speaker
  
  # Transcribe from both microphone and system audio
  python steno.py --mic --speaker
  
  # Use specific device IDs
  python steno.py --mic 1 --speaker 19
  
  # Specify output file
  python steno.py --mic --out transcript.jsonl
  
  # Adjust speaker identification threshold
  python steno.py --speaker --threshold 0.50
  
  # Verbose mode with audio meters and live transcription
  python steno.py --mic --speaker --verbose
        """
    )
    
    parser.add_argument(
        '-m', '--mic',
        nargs='?',
        const=-1,
        type=int,
        metavar='N',
        help='Transcribe from microphone. Optionally specify device ID (auto-detect if not provided)'
    )
    
    parser.add_argument(
        '-s', '--speaker',
        nargs='?',
        const=-1,
        type=int,
        metavar='N',
        help='Transcribe from system audio (speaker output). Optionally specify device ID (auto-detect if not provided)'
    )
    
    parser.add_argument(
        '-o', '--out',
        type=str,
        metavar='FILE',
        help='Output JSONL file (default: transcript.jsonl in current directory)'
    )
    
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.36,
        metavar='N',
        help='Speaker identification threshold 0.0-1.0 (default: 0.36)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print transcriptions to stdout with real-time audio level meters'
    )
    
    args = parser.parse_args()
    
    # Validate that at least one source is specified
    if args.mic is None and args.speaker is None:
        print("Error: At least one audio source (--mic or --speaker) must be specified\n")
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("=" * 60)
    print("Steno - Real-time Audio Transcription")
    print("=" * 60)
    
    # Handle microphone
    mic_device = None
    if args.mic is not None:
        if args.mic == -1:
            mic_device = auto_detect_microphone()
            if mic_device is None:
                print("Error: Could not auto-detect microphone")
                sys.exit(1)
        else:
            mic_device = args.mic
            print(f"✓ Using microphone device: {mic_device}")
    
    # Handle speaker
    speaker_device = None
    if args.speaker is not None:
        if args.speaker == -1:
            speaker_device, p_temp = auto_detect_speaker()
            if speaker_device is None:
                print("Error: Could not auto-detect speaker loopback device")
                sys.exit(1)
            if p_temp:
                p_temp.terminate()
        else:
            speaker_device = args.speaker
            p_temp = pyaudio.PyAudio()
            device_info = p_temp.get_device_info_by_index(speaker_device)
            print(f"✓ Using speaker device: [{speaker_device}] {device_info['name']}")
            p_temp.terminate()
    
    # Handle output file
    output_file = Path(args.out) if args.out else None
    
    # Create transcriber
    transcriber = AudioTranscriber(
        mic_device=mic_device,
        speaker_device=speaker_device,
        output_file=output_file,
        speaker_threshold=args.threshold,
        verbose=args.verbose
    )
    
    # Setup signal handler
    def signal_handler(sig, frame):
        transcriber.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start transcription
    try:
        transcriber.start()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
