#$ -S /bin/bash
#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=300:0:0
#$ -j y
#$ -N sgCBCTCV3in32
#$ -cwd
hostname
date
python3 -u train.py \
--project CBCTUnetSeg \
--exp_name hpc.12-3.segModeCBCT_in32_CV3 \
--data_path ../data/CBCT/fullResCropIntensityClip_resampled \
--batch_size 4 \
--input_mode cbct \
--inc 1 \
--outc 2 \
--cv 3 \
--input_shape 64 101 91 \
--lr 3e-5 \
--affine_scale 0.15 \
--save_frequency 500 \
--num_epochs 50000 \
--w_dce 1.0 \
--using_HPC 0 \
--nc_initial 32 \
--two_stage_sampling 0 \
--gpu 1
                   
                   
