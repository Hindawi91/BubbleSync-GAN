python3 main.py --mode train --dataset Boiling --crop_size 500 --image_size 256 --c_dim 1 \
                 --image_dir ./data \
                 --sample_dir Boiling/samples \
                 --source_domain DS2 --target_domain DS3 \
                 --add_blob_count_loss 1 --add_blob_mean_area_loss 1 --add_blob_std_area_loss 1\
                 --log_dir Boiling/logs \
                 --model_save_dir Boiling/models \
                 --result_dir Boiling/results \
                 --batch_size 8 --num_workers 4 --lambda_id 0.1 --num_iters 300000
 