-----------mamba_env suggest---------------

The installation of the mamba environment requires specific environment configuration. We recommend cu118+torch2.1.xxx. If the installation fails, you can directly download the .whl file for offline installation



-------------------#### train ####---------------------

Before training the dataset, you first need to change the dataset path in the file under the path "ultralytics/cfg/datasets/" to your own path. We recommend that you use an absolute path.

In order to align the training strategy with the official PaddlePaddle(RT-DETR) as much as possible, the following modifications were made to the source code:
1. Change the parameter max_norm in torch.nn.utils.clip_grad_norm_ in optimizer_step function in ultralytics/engine/trainer.py to 0.1
2. Set self.args.nbs to self.batch_size in _setup_train function in ultralytics/engine/trainer.py. The purpose of this is to make the model not need to accumulate gradients before updating parameters
3. Changes to the ultralytics/cfg/default.yaml configuration file

We put the training files of COCO dataset and VisDrone dataset into run-coco.py and run-vis.py respectively. We only need to change the file path to start training.

------------------#### test or val ####--------------

If you need to reproduce the detailed indicators in the coco dataset, including APs, etc., you need to align the file name of the trained pre_json file through the coco2json.py file. Since the naming rules of ultralytics are different from those of COCO, it cannot be run directly, so you need to run coco2json.py first. Then run get_coco_metrice.py to view detailed coco indicators

We used val to test our training results for the coco dataset, and test to test our results for the VisDrone dataset, but the batch of the tests was 1.
