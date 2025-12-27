for i in {10000..300000..10000}
  do 
     python3 main.py --mode test --dataset Boiling --crop_size 500 --image_size 256 --c_dim 1 \
                 --image_dir ./data \
                 --sample_dir Boiling/samples \
                 --log_dir Boiling/logs \
                 --model_save_dir Boiling/models \
                 --result_dir Boiling/results_$i \
                 --batch_size 1 --num_workers 4 --lambda_id 0.1 --test_iters $i
 done
