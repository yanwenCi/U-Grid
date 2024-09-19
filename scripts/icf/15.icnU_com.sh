python -u train.py --project Icn\
                   --exp_name 15.icnU16_com\
                   --ushape \
                   --cropped 0 \
                   --COM \
                   --in_nc 1 \
		             --grid_size 10 10 10 \
                   --data_path ../AS-morph-interp-ver/0.7-0.7-0.7-64-64-51 \
                   --key_file key-train-IFIB-val-IFIB-test-IFIB.pkl \
                   --gpu 0\
                   --batch_size 4 \
                   --num_epochs 400 \
                   --affine_scale 0 \
                   --lr 1e-4 \
                   --w_bde 50 \
                   --w_Idsc 10 \
                   --w_Gdsc 10 \
                   --patient_cohort inter+intra \
                   --nc_initial 16 \
                   --in_nc 1 \
                   --num_layers 4 \
      

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

