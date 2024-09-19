python -u train.py --project Icn\
	                --model ICNet_trans \
                   --cropped 0 \
                   --exp_name 06.icn_grid10_trans \
                   --data_path ../AS-morph-interp-ver/0.7-0.7-0.7-64-64-51 \
                   --key_file key-train-IFIB-val-IFIB-test-IFIB.pkl \
                   --gpu 1\
                   --num_epochs 500\
                   --batch_size 8 \
                   --input_shape 128 128 102 \
                   --affine_scale 0 \
                   --lr 1e-5 \
                   --w_bde 50 \
                   --w_Idsc 10 \
                   --w_Gdsc 10 \
                   --patient_cohort inter+intra \
                   --nc_initial 8

                #    --key_file random-key.pkl \
                #    --voxel_size 0.7 0.7 0.7 \
                #    --affine_scale 0.1 \
                #    --lr 1e-5 \
                #    --num_epochs 3000 \
                #    --ddf_energy_type bending \
                #    --w_bde 50 \
                #    --w_ssd 1.0 \
                #    --w_dce 1.0 \
                #    --w_mmd 0

