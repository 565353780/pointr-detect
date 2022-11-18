cd ../PoinTr

bash ./scripts/test.sh 0 \
    --ckpts /home/chli/chLi/PoinTr/pointr_training_from_scratch_c55_best.pth \
    --config ./cfgs/ShapeNet55_models/PoinTr.yaml \
    --mode easy \
    --exp_name example

