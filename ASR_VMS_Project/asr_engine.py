import os
import torch

class ASREngine:
    def __init__(self, backend="whisperx", model_size="large-v3", device="cuda", compute_type="float16", language="en"):
        self.backend = backend
        self.device = device if torch.cuda.is_available() else "cpu"
        self.compute_type = compute_type if self.device == "cuda" else "int8"
        self.language = language
        
        print(f"Initializing ASREngine with backend: {self.backend}, model: {model_size}, device: {self.device}")
        
        if self.backend == "whisperx":
            import whisperx
            self.model = whisperx.load_model(model_size, self.device, compute_type=self.compute_type, language=self.language)
            self.align_model, self.align_metadata = whisperx.load_align_model(language_code=self.language, device=self.device)
        elif self.backend == "faster_whisper":
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        else:
            raise ValueError(f"Unknown ASR backend: {self.backend}")

    def transcribe_words(self, audio_path, hotwords=None):
        """
        Returns a list of dictionaries:
        [{"word": str, "start": float, "end": float, "score": float}, ...]
        """
        words_out = []
        
        if self.backend == "whisperx":
            import whisperx
            audio = whisperx.load_audio(audio_path)
            
            # Note: whisperx doesn't directly support hotwords parameter in transcribe yet
            # It uses the transcription as a prior. For hotwords, we could use initial_prompt but it's limited.
            result = self.model.transcribe(audio, batch_size=8)
            
            # Align whisper output
            result_aligned = whisperx.align(result["segments"], self.align_model, self.align_metadata, audio, self.device, return_char_alignments=False)
            
            for segment in result_aligned["segments"]:
                if "words" in segment:
                    for w in segment["words"]:
                        if "start" in w and "end" in w and "word" in w and "score" in w:
                            words_out.append({
                                "word": w["word"],
                                "start": w["start"],
                                "end": w["end"],
                                "score": w["score"]
                            })
                            
        elif self.backend == "faster_whisper":
            # Using hotwords for bias
            segments, info = self.model.transcribe(audio_path, word_timestamps=True, hotwords=hotwords)
            for segment in segments:
                for w in segment.words:
                    words_out.append({
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "score": w.probability
                    })
                    
        return words_out
