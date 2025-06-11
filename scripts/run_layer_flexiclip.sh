gpu_id=0

experiment_name="layer_aniclipart"
targets=(
    # "man_breaking"
    "man_boxing2"
    "man_rap"
    # "couple_walk"
    # "couple_dance2"
    # "couple_dance"
)

for target in "${targets[@]}"; do
    output_path="output_videos_layered"
    output_folder="${target}/${experiment_name}"
    echo "==== target: $target ===="
    echo "output folder: $output_folder"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python animate_clipart.py \
        --target "svg_input/${target}/${target}" \
        --output_path "$output_path" \
        --output_folder "$output_folder" \
        --optim_bezier_points \
        --bezier_radius 0.01 \
        --augment_frames \
        --lr_bezier 0.005 \
        --num_iter 1000 \
        --num_frames 24 \
        --inter_dim 128 \
        --loop_num 2 \
        --guidance_scale 50 \
        --opt_bezier_points_with_mlp \
        --normalize_input \
        --opt_with_skeleton \
        --skeleton_weight 15 \
        --fix_start_points \
        --arap_weight 3000 \
        --opt_with_layered_arap \
        --max_tri_area 30 \
        --min_tri_degree 20 \
        --need_subdivide
done