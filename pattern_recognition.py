import sys
import csv

class pattern_analyis:
    """
        class where data is kept from the predictions + analysis functions
    """

    def __init__(self):
        self.array = []

    def add_prediction(self, selection, start, end, selection_file, sound_file):
        dict = {
            'selection': selection,
            'start': start,
            'end': end,
            'selection_file': selection_file,
            'sound_file': sound_file}
        self.array.append(dict)

    def write_to_file(self, name):
        if len(self.array) == 0:
            print('nothing to save so far!')
            return
        with open(name, 'w', newline='') as csvfile:
            fieldnames=list(self.array[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.array:
                writer.write(row)

    @staticmethod
    def read_and_analyse(filename,window, spacing):
        """
        input:
            filename: str of name of file
            window: int of how long the sliding window is
            spacing: float of how long can be in between vocalisations
        """
        pattern_dict = {}
        if filename:
            with open(filename,'r') as csvfile:
                dictreader = csv.DictReader(csvfile)
                for row in dictreader:
                    pass
    @staticmethod
    def encode(input_string):
        i = 0
        window_size = 15
        pattern_dict = {}
        while i < len(input_string):
            logged = False
            j = i + 1
            while not logged:
                cur_window = input_string[i:j]
                if cur_window not in pattern_dict.keys():
                    pattern_dict[cur_window] = len(pattern_dict)
                    logged = True
                    i = j+1
                elif j <= len(input_string):
                    j += 1
                else:
                    i = j
                    logged=True
        print(pattern_dict)
    #look at this for SW ziv https://gist.github.com/mreid/6968667

if __name__=='__main__':
    pattern_analyis.encode('ABBBBABABABABABABABSABSBA')
