import shutil
import os

for i in range(10000,300001,10000):

    # catgeories = ["post_CHF"]
    catgeories = ["pre_CHF","post_CHF"]
    trg_dir = f"./brats_syn_256_lambda0.1/results_{i}"

    for category in catgeories:
        src_dir = f"./brats_syn_256_lambda0.1/results_{i}/{category}"
        file_names = os.listdir(src_dir)
        print(len(file_names))

        for file_name in file_names:
            shutil.copy(os.path.join(src_dir, file_name), trg_dir)
            print(f'copying {file_name} from {src_dir} to {trg_dir}')
        # Delete categroy file

        shutil.rmtree(src_dir)
