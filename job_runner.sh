#!/bin/bash
#SBATCH -n 1        # number of nodes requested
#SBATCH --gres=gpu:1        # gpu you requested, change according to the node you will use                                                    
#SBATCH -C V100				# gpu name, change according to the node you will use 
#SBATCH --mem=32G			# memory you requested, change according to the node you will use 
#SBATCH -t 24:00:00 # wall time (D-HH:MM) latest HH:MM:SS, longest time you requested, your code will be terminated if it didn't finished within wall time
#SBATCH -e logs/stderr_training_job.txt # STDERR (%j = JobId), your err log output file
#SBATCH -o logs/stdout_training_job.txt # STDOUT (%j = JobId), your output log file
#SBATCH --mail-type=END # Send a notification when a job starts stops, or fails, set if you want to send yourself an email                                                
#SBATCH --mail-user=gcperkins@wpi.edu  # change to your own email address and change the below to your virtual environment setting


bash finetune.sh 1 --data /home/gcperkins/full_size_pu --model CSWin_64_12211_tiny_224 -b 64 --lr 2.5e-4 --min-lr 5e-7 --weight-decay 1e-8 --amp --img-size 224 --warmup-epochs 0 --model-ema-decay 0.9996 --finetune /home/gcperkins/CSWin-Transformer/cswin_tiny_224.pth --epochs 1000 --mixup 0.01 --cooldown-epochs 25 --interpolation bicubic  --lr-scale 0.05 --drop-path 0.2 --cutmix 0.3 --use-chk --fine-22k --ema-finetune --workers 1 --num-classes 4 --eval-metric loss

