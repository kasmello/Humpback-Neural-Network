# LZ77 Encoding
#
# Simple implementation of Lempel-Ziv 77 (a.k.a. "Sliding Window") coding
# as described in Section 13.4 of Cover & Thomas' "Information Theory".
#
# USAGE: python encode.py INPUT_STREAM
#
# EXAMPLE (from lectures):
#    $ python encode.py abbababbababbab
#    (0,a)
#    (0,b)
#    (1,1,1)
#    (1,3,2)
#    (1,5,10)
#
# AUTHOR: Mark Reid
# CREATED: 2013-10-13
import sys
import logging
#in video format: https://www.youtube.com/watch?v=cSyK2iCqr4w
###################################################################################
# Coding
#no buffer


def encode(stream):
    """Encode the input stream using a hard-coded window size"""
    window = ""     # Buffer of last MAX_WINDOW_SIZE symbols
    suffix = ""          # Suffix of stream to be coded
    i = 0           # Index into stream
    output_list=[]
    while i < len(stream):
       logging.info("----- i = " + str(i) + " | window = " + window + " | suffix = " + suffix)
       selection = stream[i]

       logging.info("READ: selection = " + selection)

       if (window + suffix).find(suffix + selection) < 0:
           # Suffix extended by x could not described by previously seen symbols.
           if suffix == "":
               # No previously seen substring so code x and add it to window
               output_list.append(code_symbol(selection))
               window = grow(window, selection)
           else:
               # Find number of symbols back that s starts
               i = lookback(window, s)
               code_pointer(i,s)
               window = grow(window, s)

               # Reset suffix and push back last symbol
               suffix = ""
               i -= 1
       else:
           # Append the last symbol to the search suffix
           suffix += selection

       # Increment the stream index
       i += 1

    # Stream finished, flush buffer
    logging.info("READ: <END>")
    if suffix != "":
        offset = lookback(window,suffix)
        output_list.append(code_pointer(offset,suffix))
    print(output_list)
def code_symbol(x):
    """Write a single symbol out"""
    return (0,x)

def code_pointer(i,s):
    """Write a pointer out"""
    return(1,i,str(len(s)))

###################################################################################
# Window management

# Hard-coded maximum window size
MAX_WINDOW_SIZE = 15

def lookback(window, s):
    """Find the lookback index for the suffix s in the given window"""
    return len(window) - (window + s).find(s) #this is first one, not closest one

def grow(window, x):
    """Update a window by adding x and keeping only MAX_WINDOW most recent symbols"""
    window += x
    if len(window) > MAX_WINDOW_SIZE:
        window = window[-MAX_WINDOW_SIZE:]   # Keep MAX_WINDOW last symbols

    return window

###################################################################################
# Main

if __name__ == "__main__":
    # Uncomment line below to show encoding state
    # logging.basicConfig(level=logging.INFO)
    input_string = input('INPUT STRING: ')
    encode(input_string)
