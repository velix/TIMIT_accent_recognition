import os
import numpy as np
from sklearn.preprocessing import StandardScaler


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

        if self.SET_NAME == 'test':
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
                                                            self.SET_NAME, 'samples')
                    mspec_path = self.io.store_in_archive(mspec, sentence,
                                                          self.SET_NAME, 'mspec')

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

    def standardize_dataset(self):
        '''
        Performs dataset-level standardization
        '''
        data = np.load(os.path.join(self.co.DATA_ROOT, '{}.npz'.format(self.SET_NAME)))
        samples = data['X']

        _, sentence_indexes_tuples = self._get_indexes_to_slice()
        data_stack = np.vstack(samples)

        scaler = StandardScaler()

        if self.SET_NAME == 'train':
            scaled = scaler.fit_transform(data_stack)

            self.TRAIN_MEAN = scaler.mean_
            self.TRAIN_VAR = scaler.var_

            filename = os.path.join(self.co.DATA_ROOT, 'training_set_meta.npz')
            np.savez(filename, mean=self.TRAIN_MEAN, var=self.TRAIN_VAR)

        elif self.SET_NAME == 'test':
            meta = np.load(os.path.join(self.co.DATA_ROOT, 'training_set_meta.npz'))
            self.TRAIN_MEAN = meta['mean']
            self.TRAIN_VAR = meta['var']

            scaled = np.divide(data_stack - self.TRAIN_MEAN, self.TRAIN_VAR)

        all_scaled_samples = []

        for slice_tuple in sentence_indexes_tuples:
            slice_start = slice_tuple[0]
            slice_end = slice_tuple[1]

            sentence = scaled[slice_start:slice_end]

            all_scaled_samples.append(sentence.tolist())

        stored_into = self._store_data(all_scaled_samples, data['Y'], data['Y_string'], 'dataset_scaled')

        return stored_into

    def standardize_speaker(self):
        '''
        Performs speaker-lever standardization
        '''
        data = np.load(os.path.join(self.co.DATA_ROOT, '{}.npz'.format(self.SET_NAME)))
        samples = data['X']

        speaker_indexes_tuples, sentence_indexes_tuples = self._get_indexes_to_slice()

        all_scaled_samples = []
        scaler = StandardScaler()

        for slice_tuple in speaker_indexes_tuples:
            slice_start = slice_tuple[0]
            slice_end = slice_tuple[1]

            # Slice the samples list, stack it to get 2d matrix
            speaker_slice = np.vstack(samples[slice_start:slice_end])
            # Scaling
            scaled = scaler.fit_transform(speaker_slice)

            # Add to the list a list of the scaled samples
            all_scaled_samples.append(scaled)

        data_stack = np.vstack(all_scaled_samples)

        all_scaled_samples = []

        for slice_tuple in sentence_indexes_tuples:
            slice_start = slice_tuple[0]
            slice_end = slice_tuple[1]

            sentence = data_stack[slice_start:slice_end]

            all_scaled_samples.append(sentence.tolist())

        stored_into = self._store_data(all_scaled_samples, data['Y'], data['Y_string'], 'speaker_scaled')

        return stored_into

    def _get_indexes_to_slice(self):
        hierarchy_filename = 'timit_{}_samples.json'.format(self.SET_NAME)

        speaker_indexes_tuples = []
        previous_speakers_sentences = 0

        sentence_indexes_tuples = []
        previous_sentences = 0

        for accent in self.io.import_from_json_lines(hierarchy_filename):
            speakers = accent['speakers']

            for speaker in speakers:
                speaker_sentences_num = len(speaker['sentences'])

                slice_start = previous_speakers_sentences
                slice_end = previous_speakers_sentences + speaker_sentences_num

                speaker_indexes_tuples.append((slice_start, slice_end))

                previous_speakers_sentences += speaker_sentences_num

                for sentence in speaker['sentences']:
                    mspec_features = np.load(sentence['mspec_path'])

                    sentence_frames = np.shape(mspec_features.T)[0]
                    slice_start = previous_sentences
                    slice_end = previous_sentences + sentence_frames

                    sentence_indexes_tuples.append((slice_start, slice_end))
                    previous_sentences += sentence_frames

        return speaker_indexes_tuples, sentence_indexes_tuples

    def _targets_to_ints(self, targets):
        '''
        Turns a list of strings to a list of unique int ids
        '''
        unique = np.unique(targets).tolist()
        out = [unique.index(t) for t in targets]

        return out

    def _store_data(self, samples, targets_int, targets, scaling_type=''):
        if scaling_type == '':
            scaling = ''
        else:
            scaling = scaling_type+'_'

        filename = os.path.join(self.co.DATA_ROOT,
                                '{}{}.npz'.format(scaling, self.SET_NAME))

        np.savez(filename, X=samples,
                 Y=targets_int, Y_string=targets)

        return filename


if __name__ == '__main__':
    preprocessor = Preprocessor('train')
    if not preprocessor.path_hierarchy_exists() or (
            not preprocessor.path_hierarchy_with_features_exists()):
        preprocessor.create_hierarchies()

    stored_into = preprocessor.transform_data()
    print("Data store in: ", stored_into)

    stored_into = preprocessor.standardize_dataset()
    print('Dataset standardized data stored into', stored_into)

    stored_into = preprocessor.standardize_speaker()
    print('Speaker standardized data stored into', stored_into)
