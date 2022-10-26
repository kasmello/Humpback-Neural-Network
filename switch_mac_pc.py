import pathlib
import pandas as pd
import platform
import openpyxl
from tqdm import tqdm

def mac_to_pc(text):
    return text.replace('/Volumes/HD/HumpbackDetect', 'D:/HumpbackDetect')

def pc_to_mac(text):
    new_text = text.replace('D:/HumpbackDetect', '/Volumes/HD/HumpbackDetect')
    new_text = new_text.replace('F:/HumpbackDetect', '/Volumes/HD/HumpbackDetect')
    new_text = new_text.replace('D:\HumpbackDetect', '/Volumes/HD/HumpbackDetect') 
    new_text = new_text.replace('\ '.strip(),'/')
    return new_text

def other_fixes(text):
    new_text = text.replace('Volume/','Volumes/')
    new_text = new_text.replace('\\','/')
    new_text = new_text.replace('/Volumes/0481098535/HumpbackDetect','/Volumes/HD/HumpbackDetect')
    new_text = new_text.replace('Begin Sample (samples)', 'Beg File Samp (samples)')
    new_text = new_text.replace('/Volumes/Karmel TestSets/', '/Volumes/KARMEL TEST/')
    new_text = new_text.replace('/Volumes/HumpbackDetect', '/Volumes/HD/HumpbackDetect')
    return new_text

def sort_all_txt(df):
    df.sort_values('Begin Time (s)',inplace=True)
    for i in range(len(df)):
        df.iloc[i]['Selection'] = i+1

    
a = {
    0: mac_to_pc,
    1: pc_to_mac,
    3: sort_all_txt,
}

if platform.system()=='Windows':
    humpback = pathlib.Path('D:/HumpbackDetect/Humpback Units/Selection Tables/').glob('*.txt')
    minke = pathlib.Path('D:/HumpbackDetect/Minke Boings/Chosen Selection Tables/').glob('*.txt')
    # minke = pathlib.Path('D:/HumpbackDetect/Minke Boings/3187 IMOS WA NW SNR4/Selection Tables/').glob('*.txt')
    # minke2 = pathlib.Path('D:/HumpbackDetect/Minke Boings/3334 1011 WAIMOS Dampier SNR007/Selection Tables/').glob('*.txt')
    
else:
    humpback = pathlib.Path('/Volumes/HD/HumpbackDetect/Humpback Units/Selection Tables/').glob('*.txt')
    minke = pathlib.Path('/Volumes/HD/HumpbackDetect/Minke Boings/Chosen Selection Tables/').glob('*.txt')
    # minke = pathlib.Path('/Volumes/HD/HumpbackDetect/Minke Boings/3187 IMOS WA NW SNR4/Selection Tables/').glob('*.txt')
    # minke2 = pathlib.Path('/Volumes/HD/HumpbackDetect/Minke Boings/3334 1011 WAIMOS Dampier SNR007/Selection Tables/').glob('*.txt')
    # testing = pathlib.Path('/Volumes/KARMEL TEST/Selection Tables/').glob('*.txt')

choosing = True
while choosing:
    choice = input('Select input:\n0: mac_to_pc\n1: pc_to_mac\n2: txt to xlsx\n3: Sort by Begin time and renumber\nOther: exit\n')
    if choice not in ['0','1','2','3']:
        print('\nBYE')
        choosing = False
    elif choice == '2':
        print('Chosen 2')
        for txt_files in [humpback, minke]:
            for file_path in txt_files:
                df = pd.read_table(file_path)
                df.to_excel(file_path.as_posix()[:-3]+'xlsx',index=False)

    elif choice == '3':
        print('Chosen 3')
        for txt_files in [humpback, minke]:
            for file_path in txt_files:
                if file_path.as_posix().split('/')[-1][:2] == '._':
                    continue
                print(file_path.as_posix())
                df = pd.read_table(file_path)
                sort_all_txt(df)
                df.to_csv(file_path.as_posix(), sep="\t", index=False)

    else:
        print(f'Chose {choice}')
        for txt_files in [humpback, minke]:
            for file_path in txt_files:
                with open(file_path, 'rb') as file:
                    all_text = file.read()
                all_text = all_text.decode('iso-8859-1')
                new_text = a[int(choice)](all_text)
                new_text = other_fixes(new_text)
                with open(file_path, 'w') as file:
                    file.write(new_text)