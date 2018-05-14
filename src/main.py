from timitManipulation import TIMIT
from dataFilesIO import DataFiles
import Utilities as u
import librosa

io = DataFiles()
tim = TIMIT()


def main():
    print('Creating path hierarchy...')
    hierarchy = tim.create_paths_hierarchy()

    print('Exporting path hierarchy...')
    io.export_to_json_lines(hierarchy, 'timit_train_path_hierarchy.json')

    print('Creating train samples...')
    for accent in io.import_from_json_lines('timit_train_path_hierarchy.json'):
        print('\tFor accent: ', accent['accent'])
        speakers = accent['speakers']

        for speaker in speakers:
            speaker_sentences = speaker['sentences']

            for sentence in speaker_sentences:
                audio = sentence['audio']

                samples, samplingrate = u.loadAudio(audio)
                mspec = librosa.feature.melspectrogram(samples,
                                                       sr=samplingrate)

                # Stores the param into an npz object
                # and return the path to the stored file
                # sentence['audio'] can give the path of the audio file for
                #   the speaker use it to extract accent, speaker and sentence
                #   id to construct a hierarchy at the end of which the archive
                #   is stored
                samples_path = io.store_in_archive(samples, sentence, 'samples')
                mspec_path = io.store_in_archive(mspec, sentence, 'mspec')

                sentence["samples_path"] = samples_path
                sentence["audio_sr"] = samplingrate
                sentence["mspec_path"] = mspec_path

        print('\t\tAttempting to store entry')
        io.export_entry_to_json_line(accent, 'timit_train_samples.json')
        print('\t\tStored')

    # io.export_to_json(accents_dir, 'timit_train_samples.json', indent=1)


if __name__ == '__main__':
    main()
