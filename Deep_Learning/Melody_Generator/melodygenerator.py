import tensorflow.keras as k
import json
import numpy as np
import music21 as m21
from preprocess import SEQ_LENGTH, MAPPING

class MelodyGenerator:

    def __init__(self, model_path='model.h5'):
        self.model_path = model_path
        self.model = k.models.load_model(model_path)
        with open(MAPPING, 'r') as fp:
            self._mappings = json.load(fp)
        self._start_symbol = ["/"] * SEQ_LENGTH

    def _sample_with_temp(self, probabilities, temperature):
        predictions = np.log(probabilities)/temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)
        return index

    def save_melody(self, melody, step_duration=0.25, format='midi', file_name='melody.midi'):
        stream = m21.stream.Stream()
        start_symbol = None
        step_counter = 1
        for i,symbol in enumerate(melody):
            if symbol != '_' or i+1 == len(melody):
                if start_symbol is not None:
                    quarter_len_dur = step_duration * step_counter
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarterLength=quarter_len_dur)
                    else:
                        m21_event = m21.note.Note(int(start_symbol),quarterLength=quarter_len_dur)
                    stream.append(m21_event)
                    step_counter=1
                start_symbol=symbol
            else:
                step_counter += 1
        stream.write(format, file_name)



    def generate(self, seed, num_steps, max_seq_length, temperature):
        seed = seed.split()
        melody = seed
        seed = self._start_symbol + seed
        seed = [self._mappings[symbol] for symbol in seed]
        for _ in range(num_steps):
            seed = seed[-max_seq_length:]
            onehot_seed = k.utils.to_categorical(seed, num_classes=len(self._mappings))
            onehot_seed = onehot_seed[np.newaxis, ...] # add an extra dimension
            probs = self.model.predict(onehot_seed)[0]
            output_int = self._sample_with_temp(probs, temperature)
            seed.append(output_int)
            output_symbol = [key for key, value in self._mappings.items() if value==output_int][0]
            if output_symbol=="/":
                break
            melody.append(output_symbol)
        return(melody)
