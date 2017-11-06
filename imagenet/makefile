IN = /home/wp01/data/ImageNet
IN = /data01/ImageNet

CVD ?= 0

train :
	date
	CUDA_VISIBLE_DEVICES=$(CVD) PYTHONPATH=.:../util_lc/Utils:../pyscatwave python imagenet/train.py --batchSize 256 --imagenetpath $(IN) --max_samples 0 --nthread 0
	date

sub :
	qsub -q gpu.q -l gpu=8 -cwd  imagenet/submit.sh


run :
	CUDA_VISIBLE_DEVICES=2 PYTHONPATH=.:../pyscatwave python imagenet/main_test.py --imagenetpath /home/wp01/data/ImageNet


ext :
	date
	CUDA_VISIBLE_DEVICES=5,6 PYTHONPATH=.:../util_lc/Utils:../pyscatwave python imagenet/extract.py --batchSize 256 --imagenetpath $(IN) --max_samples 0 --nthread 0
	date

