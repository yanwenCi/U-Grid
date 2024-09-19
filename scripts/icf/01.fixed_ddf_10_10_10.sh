python -u train.py --project Longitudinal \
                   --model LocalEncoder \
                   --exp_name 01.External_ICN_Proj_fixed_ddf_10_10_10 \
                   --data_path ../data/AS-morph-interp-ver/0.7-0.7-0.7-64-64-51 \
                   --key_file key-train-IFIB-val-IFIB-test-IFIB.pkl \
                   --gpu 0 \
                   --batch_size 8 \
                   --input_shape 128 128 102 \
                   --affine_scale 0 \
                   --lr 1e-4 \
                   --w_bde 50 \
                   --nc_initial 8 \
                   --patient_cohort inter+intra \
                   --w_mmd 0
                   

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

