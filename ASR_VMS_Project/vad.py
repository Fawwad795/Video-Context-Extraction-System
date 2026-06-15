import torch
import librosa

class VAD:
    def __init__(self, device="cpu"):
        self.device = device
        # Using torch.hub to load silero-vad
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False,
                                           trust_repo=True)
        self.get_speech_timestamps = utils[0]

    def has_speech(self, audio_path, threshold=0.3):
        """
        Loads the audio using librosa to bypass torchaudio IO errors and checks for speech.
        """
        try:
            wav, sr = librosa.load(audio_path, sr=16000)
            wav_tensor = torch.tensor(wav).float()
            
            speech_timestamps = self.get_speech_timestamps(wav_tensor, self.model, sampling_rate=16000, threshold=threshold)
            return len(speech_timestamps) > 0
        except Exception as e:
            print(f"VAD error on {audio_path}: {e}")
            # If VAD fails, assume there is speech so we don't accidentally drop chunks
            return True
