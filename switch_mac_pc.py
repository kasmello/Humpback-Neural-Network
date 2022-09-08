import pathlib
import pandas as pd
import platform
import openpyxl

def mac_to_pc(text):
    return text.replace('/Volumes/HD/HumpbackDetect', 'D:/HumpbackDetect')

def pc_to_mac(text):
    new_text = text.replace('D:/HumpbackDetect', '/Volumes/HD/HumpbackDetect')
    new_text = new_text.replace('F:/HumpbackDetect', '/Volumes/HD/HumpbackDetect')
    new_text = new_text.replace('/Volumes/HumpbackDetect', '/Volumes/HD/HumpbackDetect')
    new_text = new_text.replace('/Volumes/0481098535/HumpbackDetect','/Volumes/HD/HumpbackDetect')
    new_text = new_text.replace('D:\HumpbackDetect', '/Volumes/HD/HumpbackDetect')
    new_text = new_text.replace('\ '.strip(),'/')
    return new_text

def other_fixes(text):
    new_text = text.replace('Volume/','Volumes/')
    new_text = new_text.replace('\\','/')
    new_text = new_text.replace('Begin Sample (samples)', 'Beg File Samp (samples)')
    return new_text
    
a = {
    0: mac_to_pc,
    1: pc_to_mac,
}

if platform.system()=='Windows':
    humpback = pathlib.Path('D:/HumpbackDetect/Humpback Units/Selection Tables/').glob('*.txt')
    minke = pathlib.Path('D:/HumpbackDetect/Minke Boings/3187 IMOS WA NW SNR4/Selection Tables/').glob('*.txt')
else:
    humpback = pathlib.Path('/Volumes/HD/HumpbackDetect/Humpback Units/Selection Tables/').glob('*.txt')
    minke = pathlib.Path('/Volumes/HD/HumpbackDetect/Minke Boings/3187 IMOS WA NW SNR4/Selection Tables/').glob('*.txt')

choosing = True
while choosing:
    choice = input('Select input:\n0: mac_to_pc\n1: pc_to_mac\n2: txt to xlsx\nOther: exit\t')
    if choice != '0' and choice != '1' and choice != '2':
        print('\nBYE')
        choosing = False
    elif choice == '2':
        for txt_files in [humpback, minke]:
            for file_path in txt_files:
                df = pd.read_table(file_path)
                df.to_excel(file_path.as_posix()[:-3]+'xlsx',index=False)

    else:
        for txt_files in [humpback, minke]:
            for file_path in txt_files:
                with open(file_path, 'rb') as file:
                    all_text = file.read()
                all_text = all_text.decode('iso-8859-1')
                new_text = a[int(choice)](all_text)
                new_text = other_fixes(new_text)
                with open(file_path, 'w') as file:
                    file.write(new_text)