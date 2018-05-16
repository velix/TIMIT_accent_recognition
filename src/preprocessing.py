import os
import numpy as np

from dataFilesIO import DataFiles
from helper import Constants
from helper import Utilities
from timitManipulation import TIMIT
import librosa


class Preprocessor:

    def __init__(self):
        self.co = Constants()
        self.io = DataFiles()
        self.tim = TIMIT()
        self.u = Utilities()

    def data_exists(self):
        return os.path.exists(os.path.join(self.co.DATA_ROOT, 'train.npz'))

    def path_hierarchy_exists(self):
        return os.path.exists(os.path.join(self.co.DATA_ROOT,
                              'timit_train_path_hierarchy.json'))

    def path_hierarch_with_features_exists(self):
        return os.path.exists(os.path.join(self.co.DATA_ROOT,
                              'timit_train_samples.json'))

    def create_hierarchies(self, set_name='train'):
        print('Creating path hierarchy...')
        hierarchy = self.tim.create_paths_hierarchy()

        print('Exporting path hierarchy...')
        self.io.export_to_json_lines(hierarchy,
                                     'timit_train_path_hierarchy.json')

        print('Creating training samples and mspec features...')
        for accent in self.io.import_from_json_lines('timit_train_path_hierarchy.json'):
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
                    # and return the path to the stored file
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
            self.io.export_entry_to_json_line(accent,
                                              'timit_train_samples.json')
            print('\t\tStored')

        print('Hierarchy stores in {}'.format('timit_train_samples.json'))

    def transform_data(self, set_name='train'):
        '''
        Reads an already created directory of paths
        and loads the specified mspec features.
        Returns the utterances and their accent
        '''
        entries = self.io.import_from_json_lines(
                'timit_{}_samples.json'.format(set_name))

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

        stored_into = self._store_data('train', samples, targets_int, targets)

        return stored_into

    def _targets_to_ints(self, targets):
        '''
        Turns a list of strings to a list of unique int ids
        '''
        unique = np.unique(targets).tolist()
        out = [unique.index(t) for t in targets]

        return out

    def _store_data(self, set_type, samples, targets_int, targets):

        filename = os.path.join(self.co.DATA_ROOT, 'train.npz')
        np.savez(filename, X=samples, Y=targets_int,
                 Y_string=targets)

        return filename


if __name__ == '__main__':
    preprocessor = Preprocessor()
    if not preprocessor.path_hierarchy_exists():
        preprocessor.create_hierarchies()

    stored_into = preprocessor.transform_data()
    print("Data store in: ", stored_into)
