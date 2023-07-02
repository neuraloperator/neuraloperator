import os
import shutil

path_read  = '/home/nikola/HDD/aero-data/DataForNVR-AhmedBody/'
path_write  = '/home/nikola/HDD/ahmed/data/'

dirs = [d for d in os.listdir(path_read) if os.path.isdir(path_read + d)]


for dir in dirs:
    if dir == 'Cd_comparison':
        continue

    os.mkdir(path_write + dir)

    shutil.copyfile(path_read + dir + '/case_info.txt', path_write + dir + '/case_info.txt')

    if os.path.exists(path_read + dir + '/simpleFoam_steady/VTK'):

        shutil.copyfile(path_read + dir + '/simpleFoam_steady/VTK/simpleFoam_steady_5000/boundary/ahmed_body.vtp', 
                        path_write + dir + '/ahmed_body.vtp')
    else:
        shutil.copyfile(path_read + dir + '/simpleFoam_steady/simpleFoam_steady_5000/boundary/ahmed_body.vtp', 
                        path_write + dir + '/ahmed_body.vtp')
    
    print(dir)