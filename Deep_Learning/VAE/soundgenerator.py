from preprocess import MinMaxNormalizer
import librosa

class SoundGenerator:
    """Generates audio from Spectrograms"""

    def __init__(self, vae, hop_length):
        self.vae = vae 
        self.hop_length = hop_length
        self.min_max_normalizer = MinMaxNormalizer(0,1)
    
    def generate(self, spectrograms, min_max_values):
        generated_specs, latent_representations = self.vae.reconstruct(spectrograms)
        signals = self.convert_specs_to_audio(generated_specs, min_max_values)
        return signals, latent_representations
    
    def convert_specs_to_audio(self, spectrograms, min_max_values):
       signals = []
       for spec, min_max_value in zip(spectrograms, min_max_values):
          log_spec = spec[:,:,0]
          denorm_log_spec = self.min_max_normalizer.denormalise(log_spec,min_max_value["min"],min_max_value["max"])
          spectr = librosa.db_to_amplitude(denorm_log_spec)
          signal = librosa.istft(spectr, hop_length=self.hop_length)
          signals.append(signal)
       return signals       
    

