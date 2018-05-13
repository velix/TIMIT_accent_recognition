from timitManipulation import TIMIT
from dataFilesIO import DataFiles
import Utilities as u

io = DataFiles()
tim = TIMIT()


def main():
    print('Creating path hierarchy...')
    hierarchy = tim.create_paths_hierarchy()

    print('Exporting path hierarchy...')
    io.export_to_json_lines(hierarchy, 'timit_train_path_hierarchy.json')

    # accents_dir = io.import_from_json('timit_train_path_hierarchy.json')
    # accents_dir = hierarchy

    print('Creating train samples...')
    for accent in io.import_from_json_lines('timit_train_path_hierarchy.json'):
        print('\tFor accent: ', accent['accent'])
        speakers = accent['speakers']

        for speaker in speakers:
            # print('\t\tFor speaker: ', speaker['speaker_id'])
            speaker_sentences = speaker['sentences']

            for sentence in speaker_sentences:
                # print('\t\t\tFor sentence: ', sentence['text_type'], sentence['number'])
                audio = sentence['audio']

                samples, samplingrate = u.loadAudio(audio)

                sentence["audio_samples"] = samples.tolist()
                sentence["audio_sr"] = samplingrate

        print('\t\tAttempting to store entry')
        io.export_entry_to_json_line(accent, 'timit_train_samples.json')
        print('\t\tStored')

    # io.export_to_json(accents_dir, 'timit_train_samples.json', indent=1)


if __name__ == '__main__':
    main()