from os import listdir, rename
from os.path import isfile, join
import re 

im_csv_dir_path='/Users/sneha/Desktop/sample_outputOCT25/ch01'

pattern=re.compile('(?P<num>[^_]*)_ch(?P<channel>[^\.]*)')

channel_dict={'01': 'NucleusOverlay', '02': 'CiliaOverlay', '03': 'CentrioleOverlay'}

files = [f for f in listdir(im_csv_dir_path) if isfile(join(im_csv_dir_path, f))]


for filename in files:
    match = pattern.search(filename)
    if match:
        num = int(match.group('num'))
        channel=match.group('channel')
        src = f"{im_csv_dir_path}/{filename}"
        dest = f"{im_csv_dir_path}/{channel_dict[channel]}{num:04}.tiff"

        rename(src, dest)
