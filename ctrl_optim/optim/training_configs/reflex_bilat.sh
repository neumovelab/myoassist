#!/bin/bash
exec "$PYTHON_CMD" -m ctrl_optim.optim.train \
    --msk myolegs26 \
    --device Tutorial_L1 \
    --sim_time 20 \
    --pose_key walk_left \
    --num_strides 5 \
    --delayed 0 \
    --optim_mode single \
    --reflex_mode bilat \
    --tgt_vel 1.25 \
    --trunk_err_type ref_diff \
    --tgt_sym_th 0.1 \
    --tgt_grf_th 1.5 \
    -kine \
    --popsize 32 \
    --maxiter 1000 \
    --threads 32 \
    --sigma_gain 10 \
    --save_path results/reflex_bilat
