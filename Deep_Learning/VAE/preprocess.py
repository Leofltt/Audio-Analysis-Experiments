import librosa
import numpy as np
import os 
import pickle

class Loader:
    
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono
    
    def load(self, file_path):
        signal, sr = librosa.load(file_path, 
                              sr=self.sample_rate, 
                              duration=self.duration, 
                              mono=self.mono
                            )
        return signal

class Padder:
    
    def __init__(self, mode="constant"):
        self.mode = mode 
    
    def left_pad(self, array, num_missing_items):
        padded = np.pad(array,(num_missing_items,0),mode=self.mode)
        return padded
    
    def right_pad(self, array, num_missing_items):
        padded = np.pad(array,(0,num_missing_items),mode=self.mode)
        return padded

class LogSpec_Extractor:
    
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length
    
    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length # librosa (annoying): [1+ frame_size/2, num_frames] so we drop 1
                            ) [:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram

class MinMaxNormalizer:
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val
    
    def normalise(self, array):
        norm_array = (array - array.min()) / (array.max()-array.min())
        norm_array = norm_array * (self.max - self.min) + self.min 
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array 

class Saver:
    
    def __init__(self, fp_to_save, fp_minmax):
        self.save_dir = fp_to_save
        self.minmax_save_dir = fp_minmax 

    def save_feature(self, feature, fp):
        save_path = self._gen_save_path(fp)
        np.save(save_path, feature)
        return save_path
    
    def _gen_save_path(self,path):
        file_name = os.path.split(path)[1]
        save_path = os.path.join(self.save_dir, file_name + ".npy")
        return save_path
    
    def save_minmax_values(self, min_max_values):
       save_path = os.path.join(self.minmax_save_dir, "minmax.pkl")
       self._save(min_max_values, save_path) 
    
    @staticmethod
    def _save(data, fp):
        with open(fp, "wb") as f:
            pickle.dump(data, f)


class Preprocessing:
    """
    load the file
    pad the signal if necessary
    extract log spectrogram
    normalise it, storing the min max values for reconstruction and denorm
    save it
    """

    def __init__(self):
        self.padder = None 
        self.extractor = None 
        self.normalizer = None 
        self.saver = None 
        self.minmax_values = {}
        self._loader = None 
        self._num_samples_expected = None
    
    @property
    def loader(self):
        return self._loader
    
    @loader.setter
    def loader(self, loader):
        self._loader = loader 
        self._num_samples_expected = int(loader.sample_rate * loader.duration)

    def process(self, audio_fpath):
        for root, _, files in os.walk(audio_fpath):
            for file in files:
                fpath = os.path.join(root, file)
                print(f"{fpath}")
                self._process_file(fpath)
                print(f"Processed file {fpath}")
        self.saver.save_minmax_values(self.minmax_values)
    
    def _process_file(self,fpath):
        signal = self._loader.load(fpath)
        if self._use_padding(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_f = self.normalizer.normalise(feature)
        save_path = self.saver.save_feature(norm_f, fpath)
        self._store_minmax_values(save_path, feature.min(), feature.max())
    
    def _use_padding(self, signal):

        if len(signal) < self._num_samples_expected:
            return True 
    
    def _apply_padding(self, signal):
        num_missing_samples = self._num_samples_expected - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal
    
    def _store_minmax_values(self, fp, min, max):
        self.minmax_values[fp] = {
            "min": min,
            "max": max
        }
    


if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74  # in seconds
    SAMPLE_RATE = 22050
    MONO = True

    SPECTROGRAMS_SAVE_DIR = "/Users/leofltt/Desktop/Audio-Analysis-Experiments/Deep_Learning/VAE/spectrograms/"
    MIN_MAX_VALUES_SAVE_DIR = "/Users/leofltt/Desktop/Audio-Analysis-Experiments/Deep_Learning/VAE/minmax_values/"
    FILES_DIR = "/Users/leofltt/Desktop/Audio-Analysis-Experiments/Deep_Learning/VAE/FSDD/recordings/"

    loader = Loader(SAMPLE_RATE,DURATION,MONO)
    padder = Padder()
    extractor = LogSpec_Extractor(FRAME_SIZE,HOP_LENGTH)
    minmax_normalizer = MinMaxNormalizer(0,1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR,MIN_MAX_VALUES_SAVE_DIR)

    pipeline = Preprocessing()
    pipeline.loader = loader
    pipeline.padder = padder 
    pipeline.normalizer = minmax_normalizer
    pipeline.extractor = extractor
    pipeline.saver = saver 

    pipeline.process(FILES_DIR)