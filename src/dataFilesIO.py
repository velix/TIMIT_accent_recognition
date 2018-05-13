import json
import os
from constants import Constants


class DataFiles:

    def __init__(self):
        self.co = Constants()

    def export_to_json_lines(self, hierarchy, filename, indent=None):
        if not filename.endswith('.json'):
            raise ValueError('Filename does not end in .json')

        with open(os.path.join(self.co.DATA_ROOT, filename), 'w') as f:
            for line in hierarchy:
                json_s = json.dumps(line, indent=indent)
                f.write(json_s)
                f.write('\n')

    def import_from_json_lines(self, filename):
        if not filename.endswith('.json'):
            raise ValueError('Filename does not end in .json')

        with open(os.path.join(self.co.DATA_ROOT, filename), 'r') as f:
            for line in f:
                yield(json.loads(line))

    def export_entry_to_json_line(self, entry, filename, indent=None):
        # if not os.path.isfile(filename):
        #     with open(os.path.join(self.co.DATA_ROOT, filename), 'w') as f:
        #         log_feed = []
        #         json_s = json.dumps(log_feed, indent)
        #         f.write(json_s)

        # with open(os.path.join(self.co.DATA_ROOT, filename), 'r') as f:
        #     log_feed = json.load(f)
        #     log_feed.append(entry)

        with open(os.path.join(self.co.DATA_ROOT, filename), 'a') as f:
            json_s = json.dumps(entry, indent=indent)
            f.write(json_s)
