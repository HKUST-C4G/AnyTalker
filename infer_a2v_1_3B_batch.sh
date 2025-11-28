#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python generate_a2v_batch_multiID.py \
		--ckpt_dir="./checkpoints/Wan2.1-Fun-1.3B-Inp" \
		--task="a2v-1.3B" \
		--size="832*480" \
		--batch_gen_json="/nfs/zzzhong/codes/virtual_human/AnyTalker/input_example/good.json" \
		--batch_output="./outputs" \
		--post_trained_checkpoint_path="./checkpoints/AnyTalker/1_3B-single-v1.pth" \
		--sample_fps=24 \
		--sample_guide_scale=4.5 \
		--offload_model=True \
		--base_seed=44 \
		--dit_config="./checkpoints/AnyTalker/config_af2v_1_3B.json" \
		--det_thresh=0.15 \
		--mode="pad" \
		--use_half=True \


