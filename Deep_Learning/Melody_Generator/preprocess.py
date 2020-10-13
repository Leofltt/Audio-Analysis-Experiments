import os
import json
import music21 as m21
import tensorflow.keras as k
import numpy as np

KERN_DATASET_PATH = 'essen/europa/deutschl/erk'
SAVE_DIR = 'dataset'
FILE_NAME = 'file_ds'
MAPPING = 'mapping.json'
SEQ_LENGTH = 64
ACCEPTABLE_DUR = [0.25,0.5,0.75,1.0,1.5,2,3,4]

def load_songs_from_ds(dataset_path):
    songs = []

    # load all files from dataset with music21
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                score = m21.converter.parse(os.path.join(path, file))
                songs.append(score)
    return songs

def has_acceptable_durations(song, acc_durs):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acc_durs:
            return False
    return  True

def transpose(song):
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    if key.mode == 'major':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('C'))
    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('A'))

    transposed_song = song.transpose(interval)
    return transposed_song

def encode_song(song, time_step=0.25):
    encoded = []

    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded.append(symbol)
            else:
                encoded.append("_")
    encoded = " ".join(map(str, encoded))
    return encoded

def preprocess(dataset_path):
    print(f'loading songs...')
    songs = load_songs_from_ds(dataset_path)

    for i,song in enumerate(songs):
        if not has_acceptable_durations(song, ACCEPTABLE_DUR):
            continue
        song = transpose(song)
        encoded_song = encode_song(song)
        save_path = os.path.join(SAVE_DIR,str(i))
        with open(save_path,"w") as fp:
            fp.write(encoded_song)

def load(file_path):
    with open(file_path,'r') as fp:
        song = fp.read()
        return song

def dataset_into_one_file(dataset_path, filename_path,sequence_length):
    new_song_del = '/ ' * sequence_length
    songs = ""
    for path, _, files in os.walk(dataset_path):
        for file in files:
            fp = os.path.join(path, file)
            song = load(fp)
            songs = songs + song + " " + new_song_del
    songs = songs[:-1]
    with open(filename_path, 'w') as fp:
        fp.write(songs)
    return songs

def create_mapping(songs,mapping_path):
    mappings  = {}
    songs = songs.split()
    vocab = list(set(songs))
    for i, sym in enumerate(vocab):
        mappings[sym] = i
    with open(mapping_path,'w') as fp:
        json.dump(mappings,fp,indent=4)

def songs_to_Int(songs):
    songs_int = []
    with open(MAPPING, 'r') as fp:
        maps = json.load(fp)
    songs = songs.split()
    for sym in songs:
        songs_int.append(maps[sym])
    return songs_int

def gen_train_seqs(seq_length):
    songs = load(FILE_NAME)
    int_songs = songs_to_Int(songs)
    inputs = []
    targets = []
    seq_num = len(int_songs) - seq_length
    for i in range(seq_num):
        inputs.append(int_songs[i:i+seq_length])
        targets.append(int_songs[i+seq_length])
    # one-hot encoding
    vocab_size = len(set(int_songs))
    # inputs = (# seqs, seq len, vocab size)
    inputs = k.utils.to_categorical(inputs, num_classes=vocab_size)
    targets = np.array(targets)
    return inputs, targets
