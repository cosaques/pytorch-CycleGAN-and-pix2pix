set -ex
python test.py --dataroot ./datasets/t-shirts/ --name t-shirt-model-17-01-06-22 --model test --netG unet_256 --dataset_mode single --norm batch --gpu_ids -1
