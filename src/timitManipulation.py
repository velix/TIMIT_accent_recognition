import os
from helper import Utilities
from helper import Constants


class TIMIT:

    def __init__(self, test=False, timit_version=None):
        co = Constants()
        self.u = Utilities()

        if timit_version:
            timit_version = timit_version.lower()
            if timit_version not in ['timit', 'ffmtimit']:
                raise ValueError('Bad timit version')

            if timit_version == 'timit':
                if test:
                    self.SET_ROOT = co.TIMIT_TEST_ROOT
                else:
                    self.SET_ROOT = co.TIMIT_TRAIN_ROOT

            elif timit_version == 'ffmtimit':
                if test:
                    self.SET_ROOT = co.FFMTIMIT_TEST_ROOT
                else:
                    self.SET_ROOT = co.FFMTIMIT_TRAIN_ROOT

    def _get_accent_paths(self):
        '''
        Returns a list containing the paths to the accent directories
        in the training set, from the TIMIT_ROOT
        '''
        return [os.path.join(self.SET_ROOT, accent_dir) for accent_dir
                in os.listdir(self.SET_ROOT)]

    def _get_speaker_files(self, accent):
        '''
        accent: path to accent directory in timit
        Returns list of paths to each speaker in the given accent directory
        '''
        return [os.path.join(accent, speaker_dir) for speaker_dir
                in os.listdir(accent)]

    def _get_utterance_files(self, speaker):
        '''
        speaker: path to a speaker directory in timit
        Returns list of tuples
        Each tuple contains four paths, each path corresponds to the .wav,
        .txt, .wrd and .phn files for each sentence the speaker has uttered
        '''

        files = [os.path.join(speaker, file_type) for file_type
                 in os.listdir(speaker)]

        speaker_files = []
        count = 0
        for end in range(4, len(files), 4):
            start = count
            slice = files[start:end]
            speaker_files.append(tuple(slice))

            count += len(slice)

        return speaker_files

    def create_paths_hierarchy(self):
        '''
        Returns a list of accent dictionaries
        Each accent dictionary contains the name of the accent, the path
            to its directry and a list of speakers for the accent
        The list of speakers contains the speaker id, the gender,
            and a list of sentences the speaker has uttered
        The list of sentences contains the text type, the sentence number,
            and the paths to the phonetic, text and word transcriptions
            as well as the path to the audio file of the recording
        '''
        accent_paths = self._get_accent_paths()

        accents_list = []
        for accent_path in accent_paths:

            speaker_directories = self._get_speaker_files(accent_path)

            speakers_list = []
            for speaker in speaker_directories:
                speaker_files = self._get_utterance_files(speaker)

                speaker_dir = self._make_speaker_dic(speaker_files)

                speakers_list.append(speaker_dir)

            accent_dir = self._make_accent_dic(accent_path, speakers_list)

            accents_list.append(accent_dir)

        return accents_list

    def _make_accent_dic(self, accent_path, speakers_list):
        '''
        Creates the dictionary with info for each accent
        '''
        accent = accent_path[-3:]

        return {"accent": accent,
                "accent_path": accent_path,
                "speakers": speakers_list}

    def _make_speaker_dic(self, speaker_files):
        '''
        Creates the dictionary with info for each speaker
        '''
        _, gender, speaker_id, _, _ = self.u.path2info(speaker_files[0][0])

        sentences_list = self._make_speaker_sentences_dic(speaker_files)

        return {"speaker_id": speaker_id, "gender": gender,
                "sentences": sentences_list}

    def _make_speaker_sentences_dic(self, speaker_files):
        '''
        Creates the dictionary with info about the sentences for each speaker
        '''
        sentences = []
        for files in speaker_files:
            phoneme_transcription, text_transcription, audio, word_transcription = '', '', '', ''
            for file in files:
                if file.endswith('.phn'):
                    phoneme_transcription = file
                elif file.endswith('.txt'):
                    text_transcription = file
                elif file.endswith('.wav'):
                    audio = file
                elif file.endswith('.wrd'):
                    word_transcription = file
                else:
                    raise ValueError('File of uknown type encountered: ', file)

            _, _, _, text_type, sentence_number = self.u.path2info(files[0])

            sentence_dir = {"text_type": text_type, "number": sentence_number,
                            "phoneme_transcription": phoneme_transcription,
                            "text_transcription": text_transcription,
                            "audio": audio,
                            "word_transcription": word_transcription
                            }

            sentences.append(sentence_dir)

        return sentences

    def _is_subset(self, big, small):
        # Returns whether big contains all of small
        return all([el in big for el in small])

    def _return_ordered(self, lst1, lst2):
        '''
        Returns first the longer, and second the shorter list
        '''

        if len(lst1) <= len(lst2):
            return lst2, lst1

        return lst1, lst2

    def _merge_lists(self, lst1, lst2):
        return lst1 + lst2

    def merge_hierarchies(self, h1, h2):
        assert(len(h1) == len(h2))

        new_hierarchy = []
        for a in range(len(h1)):
            assert h1[a]['accent'] == h2[a]['accent']

            h1_speakers = h1[a]['speakers']
            h2_speakers = h2[a]['speakers']

            longer, shorter = self._return_ordered(h1_speakers, h2_speakers)

            new_speakers = []
            for speaker_dict in longer:
                speaker_id = speaker_dict['speaker_id']

                for speaker_dict_2 in shorter:
                    speaker_id_2 = speaker_dict_2['speaker_id']

                    merged_sentences = None
                    if speaker_id_2 == speaker_id:
                        merged_sentences = self._merge_lists(
                                                speaker_dict['sentences'],
                                                speaker_dict_2['sentences']
                                                )
                        continue

                new_speaker = speaker_dict
                if merged_sentences:
                    new_speaker['sentences'] = merged_sentences
                else:
                    new_speaker['sentences'] = speaker_dict['sentences']

                new_speakers.append(new_speaker)

            new_accent = h1[a]
            new_accent['speakers'] = new_speakers
            new_hierarchy.append(new_accent)

        return new_hierarchy


if __name__ == '__main__':
    tim = TIMIT()
    hier = tim.create_paths_hierarchy()