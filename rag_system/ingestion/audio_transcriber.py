"""
Audio Transcription Module using Whisper ASR via Docling
Converts audio files (MP3, WAV, M4A) to searchable text with timestamps
"""
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import os

try:
    from docling.document_converter import DocumentConverter, AudioFormatOption
    from docling.datamodel.pipeline_options import AsrPipelineOptions
    from docling.datamodel import asr_model_specs
    from docling.datamodel.base_models import InputFormat
    from docling.pipeline.asr_pipeline import AsrPipeline
    DOCLING_AUDIO_AVAILABLE = True
except ImportError:
    DOCLING_AUDIO_AVAILABLE = False
    print("⚠️ Docling audio features not available. Install: pip install docling[vlm] openai-whisper")


class AudioTranscriber:
    """
    Transcribe audio files using OpenAI Whisper Turbo via Docling.

    Features:
    - Automatic speech recognition (90+ languages)
    - Timestamp preservation for temporal context
    - Optimized Whisper Turbo model (speed + accuracy)
    - Support for MP3, WAV, M4A, FLAC formats
    """

    SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.m4a', '.flac'}

    def __init__(
        self,
        model: str = "turbo",  # Options: "turbo", "large-v3", "medium", "small"
        language: Optional[str] = None,  # Auto-detect if None
        include_timestamps: bool = True
    ):
        """
        Initialize audio transcriber.

        Args:
            model: Whisper model variant (turbo recommended for speed)
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            include_timestamps: Include timestamp markers in output
        """
        if not DOCLING_AUDIO_AVAILABLE:
            raise ImportError(
                "Docling audio support required. Install:\n"
                "pip install docling[vlm] openai-whisper"
            )

        self.model = model
        self.language = language
        self.include_timestamps = include_timestamps

        self._initialize_converter()

    def _initialize_converter(self):
        """Initialize Docling converter with Whisper ASR pipeline."""
        try:
            # Configure ASR pipeline
            pipeline_options = AsrPipelineOptions()

            # Select Whisper model
            if self.model == "turbo":
                pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO
            elif self.model == "large-v3":
                pipeline_options.asr_options = asr_model_specs.WHISPER_LARGE_V3
            else:
                # Default to turbo for best speed/accuracy balance
                pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

            # Set language if specified
            if self.language:
                pipeline_options.asr_options.language = self.language

            # Create converter with audio format option
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.AUDIO: AudioFormatOption(
                        pipeline_cls=AsrPipeline,
                        pipeline_options=pipeline_options,
                    )
                }
            )

            print(f"✅ Audio Transcriber initialized (Whisper {self.model.upper()})")

        except Exception as e:
            print(f"❌ Failed to initialize audio transcriber: {e}")
            raise

    def transcribe(
        self,
        audio_path: str,
        output_format: str = "markdown"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            output_format: Output format ("markdown" or "plain")

        Returns:
            Tuple of (transcript_text, metadata)
        """
        audio_path = Path(audio_path).resolve()

        # Validate file
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if audio_path.suffix.lower() not in self.SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported: {self.SUPPORTED_AUDIO_FORMATS}"
            )

        print(f"🎙️ Transcribing audio: {audio_path.name}")

        try:
            # Transcribe using Docling
            result = self.converter.convert(audio_path)

            # Export to desired format
            if output_format == "markdown":
                transcript = result.document.export_to_markdown()
            else:
                transcript = result.document.export_to_text()

            # Extract metadata
            metadata = {
                "source": str(audio_path),
                "format": "audio",
                "audio_format": audio_path.suffix.lower()[1:],
                "model": self.model,
                "language": self.language or "auto-detect",
                "has_timestamps": self.include_timestamps,
                "file_size_mb": audio_path.stat().st_size / (1024 * 1024)
            }

            # Parse transcript to extract timing info
            metadata.update(self._extract_timing_stats(transcript))

            print(f"✅ Transcription complete: {len(transcript)} characters")
            print(f"   Duration: ~{metadata.get('estimated_duration_seconds', 0):.0f}s")

            return transcript, metadata

        except Exception as e:
            print(f"❌ Transcription failed for {audio_path.name}: {e}")
            raise

    def _extract_timing_stats(self, transcript: str) -> Dict[str, Any]:
        """Extract timing statistics from timestamped transcript."""
        import re

        # Whisper format: [time: 0.0-4.0] Text here
        time_pattern = r'\[time: ([\d.]+)-([\d.]+)\]'
        matches = re.findall(time_pattern, transcript)

        if matches:
            # Get max end time as duration estimate
            end_times = [float(end) for _, end in matches]
            return {
                "estimated_duration_seconds": max(end_times) if end_times else 0,
                "timestamp_count": len(matches)
            }

        return {
            "estimated_duration_seconds": 0,
            "timestamp_count": 0
        }

    def transcribe_batch(
        self,
        audio_files: list[str],
        progress_callback: Optional[callable] = None
    ) -> list[Tuple[str, Dict[str, Any]]]:
        """
        Transcribe multiple audio files.

        Args:
            audio_files: List of audio file paths
            progress_callback: Optional callback(current, total)

        Returns:
            List of (transcript, metadata) tuples
        """
        results = []

        for i, audio_file in enumerate(audio_files, 1):
            try:
                transcript, metadata = self.transcribe(audio_file)
                results.append((transcript, metadata))

                if progress_callback:
                    progress_callback(i, len(audio_files))

            except Exception as e:
                print(f"⚠️ Skipping {audio_file}: {e}")
                results.append(("", {"error": str(e)}))

        return results


class TimestampedChunk:
    """Helper class for temporal chunking of transcripts."""

    @staticmethod
    def split_by_time_windows(
        transcript: str,
        window_seconds: int = 60,
        overlap_seconds: int = 10
    ) -> list[Dict[str, Any]]:
        """
        Split transcript into time-based chunks for better retrieval.

        Args:
            transcript: Timestamped transcript
            window_seconds: Chunk size in seconds
            overlap_seconds: Overlap between chunks

        Returns:
            List of chunks with temporal metadata
        """
        import re

        chunks = []
        time_pattern = r'\[time: ([\d.]+)-([\d.]+)\] (.+?)(?=\[time:|$)'
        segments = re.findall(time_pattern, transcript, re.DOTALL)

        current_chunk = []
        current_start = None
        current_end = None

        for start_str, end_str, text in segments:
            start = float(start_str)
            end = float(end_str)

            if current_start is None:
                current_start = start

            # Check if we should create new chunk
            if current_end and (end - current_start) > window_seconds:
                # Save current chunk
                chunk_text = " ".join(seg[2].strip() for seg in current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start_time": current_start,
                    "end_time": current_end,
                    "duration": current_end - current_start
                })

                # Start new chunk with overlap
                overlap_start = current_end - overlap_seconds
                current_chunk = [
                    seg for seg in current_chunk
                    if float(seg[0]) >= overlap_start
                ]
                current_start = overlap_start if current_chunk else start

            current_chunk.append((start_str, end_str, text))
            current_end = end

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(seg[2].strip() for seg in current_chunk)
            chunks.append({
                "text": chunk_text,
                "start_time": current_start,
                "end_time": current_end,
                "duration": current_end - current_start
            })

        return chunks


def test_transcriber():
    """Test audio transcriber."""
    print("\n🧪 Audio Transcriber Test")

    transcriber = AudioTranscriber(
        model="turbo",
        include_timestamps=True
    )

    print(f"Supported formats: {AudioTranscriber.SUPPORTED_AUDIO_FORMATS}")
    print("Ready to transcribe audio files!")


if __name__ == "__main__":
    test_transcriber()
