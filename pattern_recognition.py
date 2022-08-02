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
    def encode(stream):
        """
        Encode the input stream using a hard-coded window size
        """
        window = ""     # Buffer of last MAX_WINDOW_SIZE symbols
        s = ""          # Suffix of stream to be coded
        n = 0           # Index into stream
        pattern_dict = {}
        while n < len(stream):
            x = stream[n]
            
            if (window + s).find(s + x) < 0:
                # Suffix extended by x could not described by previously seen symbols.
                if s == "":
                    # No previously seen substring so code x and add it to window
                    code_symbol(x)
                    window = pattern_analysis.grow(window, x)
                else:
                    # Find number of symbols back that s starts
                    i = pattern_analysislookback(window, s)
                    pattern_analysiscode_pointer(i,s) 
                    window = pattern_analysisgrow(window, s) 

                    # Reset suffix and push back last symbol
                    s = ""
                    n -= 1
            else:
                # Append the last symbol to the search suffix 
                s += x
            
            # Increment the stream index
            n += 1

            # Stream finished, flush buffer
            if s != "":
                i = pattern_analysislookback(window,s) 
                pattern_analysiscode_pointer(i,s)

    @staticmethod
    def code_symbol(x):
        """Write a single symbol out"""
        return f'(0,{x})'

    @staticmethod
    def code_pointer(i,s):
        """Write a pointer out"""
        return f'(1,{str(i)},{str(len(s))})'

    ###################################################################################
    # Window management

    # Hard-coded maximum window size
    @staticmethod
    def lookback(window, s):
        """Find the lookback index for the suffix s in the given window"""
        return len(window) - (window + s).find(s)

    @staticmethod
    def grow(window, x):
        """Update a window by adding x and keeping only MAX_WINDOW most recent symbols"""
        window += x
        if len(window) > 5:
            window = window[-5:]   # Keep MAX_WINDOW last symbols

        return window 

    #look at this for SW ziv https://gist.github.com/mreid/6968667

if __name__=='__main__':
    pattern_analyis.encode('ABBBBABABABABABABABSABSBA')