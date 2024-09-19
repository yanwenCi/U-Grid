#$ -S /bin/bash
#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=300:0:0
#$ -j y
#$ -N segMBothCV5
#$ -cwd
hostname
date
python3 -u train.py \
--project CBCTUnetSeg \
--exp_name hpc.07-5.segModeBothCV5 \
--data_path ../data/CBCT/fullResCropIntensityClip_resampled \
--batch_size 8 \
--input_mode both \
--inc 2 \
--outc 2 \
--cv 5 \
--input_shape 64 101 91 \
--lr 3e-5 \
--affine_scale 0.15 \
--save_frequency 500 \
--num_epochs 50000 \
--w_dce 1.0 \
--using_HPC 1 \
--nc_initial 16 \
--two_stage_sampling 0 
                   
                   