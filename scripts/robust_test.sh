#!/usr/bin/env bash
set -x
DATAPATH="/home/luca/Scrivania/Universita/Dottorato/vpp_all/vpp/datasets/kitti2015/"
CUDA_VISIBLE_DEVICES=0 python robust_test.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti15_errortest.txt --batch_size 4 --test_batch_size 2 \
    --testlist ./filenames/kitti15_errortest.txt --maxdisp 256 \
    --epochs 1 --lr 0.001  --lrepochs "300:10" \
    --loadckpt "/home/luca/Scrivania/Universita/Dottorato/vpp_all/vpp/thirdparty/CFNet/checkpoints/sceneflow_pretraining.ckpt" \
    --model cfnet --logdir ./checkpoints/robust_abstudy_test
