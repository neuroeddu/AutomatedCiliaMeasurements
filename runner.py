import subprocess 
import sys
import os

SCRIPT_PATH=sys.path[0]
CHANNEL_DICT = {
        "01": "NucleusOverlay",
        "02": "CiliaOverlay",
        "03": "CentrioleOverlay",
    }

if sys.version_info.major != 3 or sys.version_info.minor != 9:
    print('incorrect python version. this script needs python 3.9 to be installed.')
print('checking if poetry is installed')
result = subprocess.run(['poetry', '-v'], capture_output=True)

subprocess.run(['cd', SCRIPT_PATH])

# TODO poetry install
if result.returncode:
    print('poetry not found. installing poetry')

result = subprocess.run(['poetry', 'show', '--no-dev'], capture_output=True, text=True)
installed_pkg = result.stdout

if '!' in installed_pkg:
    print('installing necessary packages')
    subprocess.run(['poetry', 'install', '--no-dev'])

# TODO Make sub-folders inside directory specified for all results 
dir_out=input('ready to start! where should everything go? ')
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

csvs_in=input('what is the path to your cellprofiler output csvs? ')
images_in=input('what is the path to your cellprofiler output images? ')

print(sys.argv[0])
1/0

print('running center2center script')
c2c_output_path = os.path.join(dir_out, 'c2c_output')

if not os.path.exists(c2c_output_path):
    os.mkdir(c2c_output_path)

subprocess.run(['poetry', 'run', 'center2center.py', '-i', csvs_in, '-o', c2c_output_path])

# CLUSTERING 
clustering=input('would you like to run clustering? y/n ')
if clustering=='y':
    subprocess.run(['poetry', 'run', 'clustering.py', '-m', os.path.join(csvs_in, 'MyExpt_Cilia.csv'), '-c', os.path.join(c2c_output_path, 'new_cilia.csv')])

# CELLPROFILER VISUALIZER
visualize_cprof=input('would you like to visualize the results of the cellprofiler pipeline? y/n')
if visualize_cprof=='y':
    num=input('enter number of images you want to visualize; if all images, press ENTER') or None
    centriole=input('enter YES if you want centriole visualization to be included. if not, press ENTER.') or None
    cprof_vis_output = os.path.join(dir_out, 'cprof_vis_output')

    if not os.path.exists(cprof_vis_output):
        os.mkdir(cprof_vis_output)
    
    subprocess.run(['poetry', 'run', 'label_cprof_im.py', '-i', csvs_in, '-m', images_in, '-o', cprof_vis_output, '-n', num, '-c', centriole])

# DATA TABLE 
data_tbl = input('would you like to make a data table? y/n')
if data_tbl=='y':
    subprocess.run(['poetry', 'run', 'data_table.py', '-i', csvs_in, '-c', c2c_output_path])

# LABEL C2C 
c2c_vis = input('would you like to visualize the c2c pipeline? y/n')
if c2c_vis=='y':
    num=input('enter number of images you want to visualize; if all images, press ENTER') or None
    
    c2c_vis_output = os.path.join(dir_out, 'c2c_vis_output')

    if not os.path.exists(c2c_vis_output):
        os.mkdir(c2c_vis_output)
    
    subprocess.run(['poetry', 'run', 'c2c_vis_output.py', '-i', csvs_in, '-m', images_in, '-o', c2c_vis_output, '-n', num, '-c', c2c_output_path])


# LABEL CILIA
organelle_label=input('would you like to label one organelle type? y/n')
if organelle_label=='y':
    channel=input('input channel you want to label: 01 for nuclei, 02 for cilia, 03 for centriole')
    num=input('enter number of images you want to visualize; if all images, press ENTER') or None

    channel_lbl = os.path.join(dir_out, f'channel_{channel}_vis_output')

    if not os.path.exists(channel_lbl):
        os.mkdir(channel_lbl)

    subprocess.run(['poetry', 'run', 'label_valid_cilia.py', '-m', csvs_in, '-g', images_in, '-o', channel_lbl, '-n', num, '-c', c2c_output_path, '-h', channel])


# CHECK ACCURACY
accuracy = input('would you like to run check_accuracy? y/n')
if accuracy=='y':
    true_results = input('enter the path to the true results')
    accuracy_path = os.path.join(dir_out, 'check_accuracy')

    if not os.path.exists(accuracy_path):
        os.mkdir(accuracy_path)

    subprocess.run(['poetry', 'run', 'check_accuracy.py', '-t', true_results, '-c', os.path.join(c2c_output_path, 'new_cilia.csv'), '-o', accuracy_path])



