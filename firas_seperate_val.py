import shutil
import os

for i in range(10000,300001,10000):
    print(i)
    catgeories = ["pre_CHF","post_CHF"]
    
    src_dir = f"./brats_syn_256_lambda0.1/results_{i}"
    
    GIT_dir = f"./brats_syn_256_lambda0.1/results_{i}"
    # UT_dir = f"/home/local/ASUAD/falhinda/Downloads/Fixed-Point-GAN-2/brats_syn_256_lambda0.1/results_formatted/UT"
    
    
    
    file_names = os.listdir(src_dir)
    print(len(file_names))
    
    
    file_names = os.listdir(src_dir)
    print(len(file_names))
    
    
    for file_name in file_names:

        if "CHF" in file_name:
            sub_dir = f'{GIT_dir}/{catgeories[1]}'
        else:
            sub_dir = f'{GIT_dir}/{catgeories[0]}'
            
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        shutil.copy(os.path.join(src_dir, file_name), sub_dir)
        print(f'copying {file_name} from {src_dir} to {sub_dir}')
        # Delete image after moving it to categroy file
        if file_name.endswith('.jpg'):
            os.remove((os.path.join(src_dir, file_name)))
