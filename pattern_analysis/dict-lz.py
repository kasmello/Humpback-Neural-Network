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
    encode('ABBBBABABABABABABABSABSBA')
