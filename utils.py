import numpy as np      
import matplotlib.pyplot as plt 
import scipy.io.wavfile 
import subprocess
import librosa
import librosa.display
import IPython.display as ipd

from pathlib import Path, PurePath   
from tqdm.notebook import tqdm
import os
import pandas as pd
import numpy as np

def convert_mp3_to_wav(audio:str) -> str:  
    """Convert an input MP3 audio track into a WAV file.

    Args:
        audio (str): An input audio track.

    Returns:
        [str]: WAV filename.
    """
    if audio[-3:] == "mp3":
        wav_audio = audio[:-3] + "wav"
        if not Path(wav_audio).exists():
                subprocess.check_output(f"ffmpeg -i {audio} {wav_audio}", shell=True)
        return wav_audio
    
    return audio

def plot_spectrogram_and_picks(track:np.ndarray, sr:int, peaks:np.ndarray, onset_env:np.ndarray) -> None:
    """[summary]

    Args:
        track (np.ndarray): A track.
        sr (int): Aampling rate.
        peaks (np.ndarray): Indices of peaks in the track.
        onset_env (np.ndarray): Vector containing the onset strength envelope.
    """
    times = librosa.frames_to_time(np.arange(len(onset_env)),
                            sr=sr, hop_length=HOP_SIZE)

    plt.figure()
    ax = plt.subplot(2, 1, 2)
    D = librosa.stft(track)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max),
                            y_axis='log', x_axis='time')
    plt.subplot(2, 1, 1, sharex=ax)
    plt.plot(times, onset_env, alpha=0.8, label='Onset strength')
    plt.vlines(times[peaks], 0,
            onset_env.max(), color='r', alpha=0.8,
            label='Selected peaks')
    plt.legend(frameon=True, framealpha=0.8)
    plt.axis('tight')
    plt.tight_layout()
    plt.show()

def load_audio_peaks(audio, offset, duration, hop_size):
    """Load the tracks and peaks of an audio.

    Args:
        audio (string, int, pathlib.Path or file-like object): [description]
        offset (float): start reading after this time (in seconds)
        duration (float): only load up to this much audio (in seconds)
        hop_size (int): the hop_length

    Returns:
        tuple: Returns the audio time series (track) and sampling rate (sr), a vector containing the onset strength envelope
        (onset_env), and the indices of peaks in track (peaks).
    """
    try:
        track, sr = librosa.load(audio, offset=offset, duration=duration)
        onset_env = librosa.onset.onset_strength(track, sr=sr, hop_length=hop_size)
        peaks = librosa.util.peak_pick(onset_env, 10, 10, 10, 10, 0.5, 0.5)
    except Error as e:
        print('An error occurred processing ', str(audio))
        print(e)

    return track, sr, onset_env, peaks

def create_matrix_signature(number_of_permutations, peaks_df):
    """
    Returns a tuple where the first element is the matrix_signature, and the second one is the matrix of permutations

            Parameters:
                    number_of_permutations (int): number of times that we will make the permutation 
                    peaks_df (dataframe): dataframe (number shingles x number document) and must contain 1 in the [shingles, document] cell if the shingles is inside the document 

            Returns:
                    matrix_signature (matrix): signature matrix, where it will have as many rows as there are permutations 
                    permutations (matrix): contains the order of the indices for each permutation 

    """
    permutations = []
    matrix_signature = []

    for i in range(number_of_permutations):
        positions_of_first_one = []
        peaks_df = peaks_df.sample(frac=1) # permutation of rows
        permutations.append(np.array(peaks_df.index))

        for song_id in peaks_df.columns:
            positions_of_first_one.append(np.argmax(peaks_df[song_id])) # add first occurence of 1 in the column
        matrix_signature.append(np.array(positions_of_first_one))
        
    return np.array(matrix_signature), permutations

def create_buckets(matrix_signature, b, permutations):
    """
    Returns a dict which contain all buckets created

            Parameters:
                    matrix_signature (matrix)
                    r_bin(integer)

            Returns:
                    bucket (dict): as a key we will have the hash and as a value we will have the documents containing this hash 

    """
    buckets = dict()
    r = int(len(permutations) / b) # number element for each hash
    # for each column insert bucket
    for index_col, col in enumerate(matrix_signature.T):
        for i in range(0, len(col), r): # we choise b = 2 
            hash = tuple(col[i:i+r])
            if hash in buckets:
                buckets[hash].add(str(index_col))
            else:
                buckets[hash] = {str(index_col)}
    return buckets

def create_buckets_query(permutations, peaks_df, query_peaks, b):
    """
    Returns a tuple where the first element is the matrix_signature, and the second one is the matrix of permutations

            Parameters:
                    permutations (matrix): contains the order of the indices for each permutation
                    peaks_df (dataframe): dataframe (number shingles x number document) and must contain 1 in the [shingles, document] cell if the shingles is inside the document 
                    query_peaks (list): list of peaks of query 
                    b(integer): number hash for query

            Returns:
                    buckets_query (list): contains the bucket choises for this query 

    """
    r = int(len(permutations) / b) # number element for each hash
    peaks_df['query'] = 0
    peaks_df.loc[query_peaks, 'query'] = 1
    hash_query = []
    for perm in permutations:
        hash_query.append(np.argmax(peaks_df.loc[perm, "query"]))

    buckets_query = set()
    for i in range(0,len(hash_query),r):
        buckets_query.add(tuple(hash_query[i:i+r]))
    
    peaks_df.drop("query", axis=1, inplace=True)
    return buckets_query

def find_matches(query, buckets_query, buckets, THRESHOLD):
    """
    Print the find matches, depending of the threshold give in input

            Parameters:
                    query(string)
                    buckets_query (list): contains the bucket choises for this query  
                    bucket (dict): as a key we will have the hash and as a value we will have the documents containing this hash 
                    THRESHOLD(float) number between 0 and 1

    """
    # doc_to_peaks contain as key all matches doc of query, and as value the list of peak
    doc_to_peaks = dict()
    for bucket in buckets_query:
        if bucket in buckets:
            for doc in buckets[bucket]:
                if doc not in doc_to_peaks:
                    doc_to_peaks[doc] = [bucket]
                else:
                    doc_to_peaks[doc].append(bucket)
    # find best matches where we go to count the number occurence of doc divided by the number of hash of query
    diz = dict()
    for doc in doc_to_peaks:
        similarity = len(doc_to_peaks[doc]) / len(buckets_query)
        if similarity >= THRESHOLD:
            diz[df_song[df_song["song_id"] == int(doc)]['song_name'].to_string(index=False)] = similarity
    for k in diz:
        print("{:<20} {:<60} {:<80}".format("",k, diz[k]))
    

def find_best_match(buckets_query, buckets, dataframe, column):
    """
    return the best match

            Parameters:
                    buckets_query (list): contains the bucket choises for this query  
                    bucket (dict): as a key we will have the hash and as a value we will have the documents containing this hash 
                    dataframe
                    column(String) column where we go to find the best match
            Returns:
                    best match

    """
    doc_result = []
    for bucket in buckets_query:
        if bucket in buckets:
            doc_result.extend(buckets[bucket])
    return dataframe[dataframe[column] == int(max(doc_result, key=doc_result.count))]

def inerzia(x,y):
    return np.sum((x - y)**2)

def distance_from_centroids(cluster_to_unit, centroids): 
    distance = 0
    for cluster in cluster_to_unit:
        distance += inerzia(centroids[cluster], cluster_rows[cluster])
    return distance

def euclidian_distance(centroids, y):
    return np.sqrt(np.sum((centroids - y)**2,axis=1))
