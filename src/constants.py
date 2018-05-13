import os


class Constants:
    def __init__(self):

        self.ROOT = '..'
        self.DATA_ROOT = os.path.join(self.ROOT, 'data')
        self._create_data_root()

        self.TIMIT_ROOT = os.path.join(self.ROOT, 'timit', 'timit')
        self.TRAIN_ROOT = os.path.join(self.TIMIT_ROOT, 'train')
        self.TEST_ROOT = os.path.join(self.TIMIT_ROOT, 'test')

    def _create_data_root(self):
        if not os.path.exists(self.DATA_ROOT):
            os.mkdir(self.DATA_ROOT)
