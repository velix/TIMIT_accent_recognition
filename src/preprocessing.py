import os
import numpy as np

from dataFilesIO import DataFiles
from helper import Constants
from helper import Utilities
from timitManipulation import TIMIT
import librosa


class Preprocessor:

    def __init__(self, set_name):
        self.SET_NAME = set_name
        self.co = Constants()
        self.io = DataFiles()

        if set_name == 'test':
            self.tim = TIMIT(test=True)
        else:
            self.tim = TIMIT()

        self.u = Utilities()

    def data_exists(self):
        filename = '{}.npz'.format(self.SET_NAME)
        return os.path.exists(os.path.join(self.co.DATA_ROOT, filename))

    def path_hierarchy_exists(self):
        filename = 'timit_{}_path_hierarchy.json'.format(self.SET_NAME)
        return os.path.exists(os.path.join(self.co.DATA_ROOT, filename))

    def path_hierarchy_with_features_exists(self):
        filename = 'timit_{}_samples.json'.format(self.SET_NAME)
        return os.path.exists(os.path.join(self.co.DATA_ROOT, filename))

    def create_hierarchies(self):
        print('Creating path hierarchy...')
        hierarchy = self.tim.create_paths_hierarchy()

        print('Exporting path hierarchy...')
        hierarchy_filename = 'timit_{}_path_hierarchy.json'.format(self.SET_NAME)
        self.io.export_to_json_lines(hierarchy, hierarchy_filename)

        print('Creating training samples and mspec features...')
        for accent in self.io.import_from_json_lines(hierarchy_filename):
            print('\tFor accent: ', accent['accent'])
            speakers = accent['speakers']

            for speaker in speakers:
                speaker_sentences = speaker['sentences']

                for sentence in speaker_sentences:
                    audio = sentence['audio']

                    samples, samplingrate = self.u.loadAudio(audio)
                    mspec = librosa.feature.melspectrogram(samples,
                                                           sr=samplingrate)

                    # Stores the param into an npz object
                    # and returns the path to the stored file.
                    # sentence['audio'] can give the path of the audio file for
                    #   the speaker use it to extract accent, speaker and
                    #   sentence id to construct a hierarchy at the end of
                    #   which the archive is stored
                    samples_path = self.io.store_in_archive(samples, sentence,
                                                            'samples')
                    mspec_path = self.io.store_in_archive(mspec, sentence,
                                                          'mspec')

                    sentence["samples_path"] = samples_path
                    sentence["audio_sr"] = samplingrate
                    sentence["mspec_path"] = mspec_path

            print('\t\tAttempting to store entry')
            samples_filename = 'timit_{}_samples.json'.format(self.SET_NAME)
            self.io.export_entry_to_json_line(accent, samples_filename)
            print('\t\tStored')

        print('Hierarchy stored in {}'.format(hierarchy_filename))

    def transform_data(self):
        '''
        Reads an already created directory of paths
        and loads the specified mspec features.
        Returns the utterances and their accent
        '''
        entries = self.io.import_from_json_lines(
                'timit_{}_samples.json'.format(self.SET_NAME))

        samples, targets = [], []

        for entry in entries:
            accent = entry['accent']
            for speaker in entry['speakers']:
                for sentence in speaker['sentences']:
                    # The mspec features are 128*frames
                    mspec_features = np.load(sentence['mspec_path'])
                    # Transpose them to frames*128
                    samples.append(mspec_features.T)
                    # Each accent is a string in dri where i =[1, 8]
                    targets.append(accent)

        targets_int = self._targets_to_ints(targets)

        stored_into = self._store_data(samples, targets_int, targets)

        return stored_into

    def _targets_to_ints(self, targets):
        '''
        Turns a list of strings to a list of unique int ids
        '''
        unique = np.unique(targets).tolist()
        out = [unique.index(t) for t in targets]

        return out

    def _store_data(self, samples, targets_int, targets):

        filename = os.path.join(self.co.DATA_ROOT, '{}.npz'.format(self.SET_NAME))
        np.savez(filename, X=samples, Y=targets_int,
                 Y_string=targets)

        return filename


if __name__ == '__main__':
    preprocessor = Preprocessor('test')
    if not preprocessor.path_hierarchy_exists() or (
            not preprocessor.path_hierarchy_with_features_exists()):
        preprocessor.create_hierarchies()

    stored_into = preprocessor.transform_data()
    print("Data store in: ", stored_into)
