import os
import json
import numpy as np
from pysndfile import sndio
from scipy.fftpack import fft
import scipy.signal as signal

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

        self.PAD_SIZE = 100
        self.L2_REG_RATE = 0.01
        self.DROPOUT_RATE = 0.3
        self.OUTPUT_SIZE = 8
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 32
        self.EPOCHS = 10

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

    def enframe(self, samples, winlen, winshift):
        """
        Slices the input samples into overlapping windows.

        Args:
            winlen: window length in samples.
            winshift: shift of consecutive windows in samples
        Returns:
            numpy array [N x winlen], where N is the number of windows that fit
            in the input signal
        """
        # The window length is sampling_rate*window_length_in_ms
        length = len(samples)
        start_indices = np.arange(0, length, winshift)
        end_indices = np.arange(winlen, length, winlen - winshift)
        pairs = zip(start_indices, end_indices)

        output = [samples[i[0]: i[1]] for i in pairs]

        # myplot(output, 'Framing')

        return output

    def preemp(self, input, p=0.97):
        """
        Pre-emphasis filter.

        Args:
            input: array of speech frames [N x M] where N is the number
                of frames and M the samples per frame
            p: preemhasis factor

        Output:
            output: array of pre-emphasised speech samples
        """

        b = np.array([1., -p])
        a = np.array([1.])

        output = signal.lfilter(b, a, input, axis=1)

        # myplot(output, 'pre-emphasis')

        return output

    def windowing(self, input):
        """
        Applies hamming window to the input frames.

        Args:
            input: array of speech samples [N x M] where N is the
                number of frames and M the samples per frame
        Output:
            array of windowed speech samples [N x M]
        """
        N, M = np.shape(input)

        window = signal.hamming(M, sym=0)

        window_axis = lambda sample: sample * window

        output = np.apply_along_axis(window_axis, 1, input)

        # myplot(output, 'Hamming Window')

        return output

    def powerSpectrum(self, input, nfft):
        """
        Calculates the power spectrum of the input signal, that is the
        square of the modulus of the FFT

        Args:
            input: array of speech samples [N x M] where N is the 
                number of frames and M the samples per frame
            nfft: length of the FFT
        Output:
            array of power spectra [N x nfft]
        Note: you can use the function fft from scipy.fftpack
        """
        result = fft(input, nfft)
        result = np.power(np.absolute(result), 2)

        # myplot(result, 'Power Spectogram')

        return result

    def get_power_spectrum(self, samples, winlen=400, winshift=200, preempcoeff=0.97,
                           nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):

        frames = self.enframe(samples, winlen, winshift)
        preemph = self.preemp(frames, preempcoeff)
        windowed = self.windowing(preemph)
        spec = self.powerSpectrum(windowed, nfft)

        return spec

    def logMelSpectrum(self, input, samplingrate):
        """
        Calculates the log output of a Mel filterbank when the input is the power spectrum

        Args:
            input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
                nfft the length of each spectrum
            samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
        Output:
            array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
            of filters in the filterbank
        Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
            nmelfilters
        """
        nfft = input.shape[1]
        N = input.shape[0]
        # filters: [N, nfft]
        filters = self.trfbank(samplingrate, nfft)

        # plot Mel filters
        # plt.plot(filters)
        # plt.title('Mel filters')
        # plt.show()

        output = np.zeros((N, filters.shape[0]))
        for j in range(filters.shape[0]):  # apply each filterbank to the whole power spectrum
            for i in range(N):
                output[i, j] = np.log(np.sum(input[i] * filters[j]))

        # myplot(output, 'Filter Banks')

        return output

    def trfbank(self, fs, nfft, lowfreq=133.33, linsc=200/3., logsc=1.0711703,
                nlinfilt=13, nlogfilt=27, equalareas=False):
        """Compute triangular filterbank for MFCC computation.

        Inputs:
        fs:         sampling frequency (rate)
        nfft:       length of the fft
        lowfreq:    frequency of the lowest filter
        linsc:      scale for the linear filters
        logsc:      scale for the logaritmic filters
        nlinfilt:   number of linear filters
        nlogfilt:   number of log filters

        Outputs:
        res:  array with shape [N, nfft], with filter amplitudes for each column.
                (N=nlinfilt+nlogfilt)
        From scikits.talkbox"""
        # Total number of filters
        nfilt = nlinfilt + nlogfilt

        #------------------------
        # Compute the filter bank
        #------------------------
        # Compute start/middle/end points of the triangular filters in spectral
        # domain
        freqs = np.zeros(nfilt+2)
        freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
        freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
        if equalareas:
            heights = np.ones(nfilt)
        else:
            heights = 2./(freqs[2:] - freqs[0:-2])

        # Compute filterbank coeff (in fft domain, in bins)
        fbank = np.zeros((nfilt, nfft))
        # FFT bins (in Hz)
        nfreqs = np.arange(nfft) / (1. * nfft) * fs
        for i in range(nfilt):
            low = freqs[i]
            cen = freqs[i+1]
            hi = freqs[i+2]

            lid = np.arange(np.floor(low * nfft / fs) + 1,
                            np.floor(cen * nfft / fs) + 1, dtype=np.int)
            lslope = heights[i] / (cen - low)
            rid = np.arange(np.floor(cen * nfft / fs) + 1,
                            np.floor(hi * nfft / fs) + 1, dtype=np.int)
            rslope = heights[i] / (hi - cen)
            fbank[i][lid] = lslope * (nfreqs[lid] - low)
            fbank[i][rid] = rslope * (hi - nfreqs[rid])

        return fbank

    def get_mspec(self, samples, winlen=400, winshift=200, preempcoeff=0.97,
                  nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):

        frames = self.enframe(samples, winlen, winshift)
        preemph = self.preemp(frames, preempcoeff)
        windowed = self.windowing(preemph)
        spec = self.powerSpectrum(windowed, nfft)
        mspec = self.logMelSpectrum(spec, samplingrate)

        return mspec


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
