#$ -S /bin/bash
#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=96:0:0
#$ -j y
#$ -N p48bs32
#$ -cwd
hostname
date

python3 -u train.py --project Longitudinal \
                   --model LocalModel \
                   --exp_name 02.External_ICN_Proj_patch_48_bs_32 \
                   --data_path ../data/AS-morph-interp-ver-ldmk/0.7-0.7-0.7-64-64-51 \
                   --key_file key-train-IFIB-val-IFIB-test-IFIB.pkl \
                   --gpu 0 \
                   --batch_size 16 \
                   --input_shape 128 128 102 \
		           --patched 1 \
		           --patch_size 48 48 48 \
                   --affine_scale 0 \
                   --lr 3e-5 \
                   --w_bde 50 \
                   --nc_initial 16 \
                   --patient_cohort inter+intra \
                   --save_frequency 50 \
                   --num_epochs 10000 \
                   --using_HPC 1 \
                   --w_mmd 0
                   



