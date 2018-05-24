import os
import json
import numpy as np
from pysndfile import sndio


class Constants:
    def __init__(self):

        self.ROOT = '..'
        self.DATA_ROOT = os.path.join(self.ROOT, 'data')
        self._create_dir(self.DATA_ROOT)

        self.LOG_ROOT = os.path.join(self.ROOT, 'logs')
        self.MODELS_ROOT = os.path.join(self.ROOT, 'models')
        self._create_dir(self.MODELS_ROOT)

        self.TIMIT_ROOT = os.path.join(self.ROOT, 'timit', 'timit')
        self.TRAIN_ROOT = os.path.join(self.TIMIT_ROOT, 'train')
        self.TEST_ROOT = os.path.join(self.TIMIT_ROOT, 'test')
        self.FIG_ROOT = os.path.join(self.ROOT, 'fig')
        self._create_dir(self.FIG_ROOT)

        self.PAD_SIZE = 244
        self.L2_REG_RATE = 0.00001
        self.DROPOUT_RATE = 0.5
        self.OUTPUT_SIZE = 8
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 64
        self.EPOCHS = 1000

    def _create_dir(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)

    def net_params_to_dictionary(self):
        return {
            'EPOCHS': self.EPOCHS,
            'LEARNING_RATE': self.LEARNING_RATE,
            'BATCH_SIZE': self.BATCH_SIZE,
            'PAD_SIZE': self.PAD_SIZE,
            'DROPOUT_RATE': self.DROPOUT_RATE,
            'L2_REG_RATE': self.L2_REG_RATE,
            'OUTPUT_SIZE': self.OUTPUT_SIZE,

        }


class Utilities:
    def __init__(self):
        pass

    def path2info(self, path):
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

    def loadAudio(self, filename):
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


class Logging:

    def __init__(self, log_name='log.json'):
        self.co = Constants()
        self.LOG_ROOT = self.co.LOG_ROOT
        self.LOG_NAME = os.path.join(self.LOG_ROOT, log_name)

        self.co._create_dir(self.LOG_ROOT)
        self._log_creation(self.LOG_NAME)

    def _log_dir_creation(self, LOG_ROOT):
        if not os.path.exists(LOG_ROOT):
            os.mkdir(LOG_ROOT)

    def _log_creation(self, LOG_NAME):
        if not os.path.isfile(LOG_NAME):
            with open(LOG_NAME, "w") as f:
                log_feed = []
                json_p = json.dumps(log_feed)
                f.write(json_p)

    def store_log_entry(self, entry):
        if not type(entry) == dict:
            raise ValueError('Store dictionary entries, not ',
                             type(entry))

        with open(self.LOG_NAME, "r") as f:
            log_feed = json.load(f)
            log_feed.append(entry)

        with open(self.LOG_NAME, "w") as f:
            json_p = json.dumps(log_feed, indent=3)
            f.write(json_p)

    def read_log(self):
        with open(self.LOG_NAME, 'r') as f:
            log_feed = json.load(f)
            # If the above line does not work, comment and use the below
            # log_feed = json.loads(f)

        return log_feed
