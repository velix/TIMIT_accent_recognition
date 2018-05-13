import numpy as np
import os
from pysndfile import sndio


def path2info(path):
    """
    path2info: parses paths in the TIMIT format and extracts information
               about the speaker and the utterance

    Example:
    path2info('../timit/timit/train/dr1/fcjf0/sa1.wav')
    """
    rest, filename = os.path.split(path)
    rest, speaker = os.path.split(rest)
    gender = speaker[0]
    speaker_id = speaker[1:]
    rest, accent = os.path.split(rest)
    utterance = filename[:-4]
    text_type = utterance[:2]
    sentence_number = utterance[2:]
    return accent, gender, speaker_id, text_type, sentence_number


def loadAudio(filename):
    """
    loadAudio: loads audio data from file using pysndfile

    Note that, by default pysndfile converts the samples into floating point
    numbers and rescales them in the range [-1, 1]. This can be avoided by
    specifying the dtype argument in sndio.read(). However, when I imported'
    the data in lab 1 and 2, computed features and trained the HMM models,
    I used the default behaviour in sndio.read() and rescaled the samples
    in the int16 range instead. In order to compute features that are
    compatible with the models, we have to follow the same procedure again.
    This will be simplified in future years.
    """
    sndobj = sndio.read(filename)
    samplingrate = sndobj[1]
    samples = np.array(sndobj[0])*np.iinfo(np.int16).max
    return samples, samplingrate
