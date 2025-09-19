# ACL-SV
This repository contains official pytorch implementation for “ADDITIVE CONSISTENCY LEARNING FOR ROBUST SPEAKER VERIFICATION IN NOISY ENVIRONMENTS” paper.


# Abstract
![overall](https://github.com/user-attachments/assets/631920ef-091b-4b16-92d1-8e26bfd9f358)

Speaker verification (SV) systems in real-world conditions often suffer performance degradation due to diverse noise.
A common mitigation is to incorporate a speech enhancement (SE) module as a frontend.
Such existing studies typically train the SE module to reconstruct predefined ``clean'' references, but these references still contain remaining noise and channel variability.
Consequently, conventional learning strategy may also restore these non-discriminative factors, degrading SV performance.
To address this limitation, we propose an Additive Consistency Learning–based Speaker Verification (ACL-SV) system.
Unlike reference-dependent training, ACL-SV ensures that the sum of the separated speaker and background representations is identical to the original input mixture.
This ensures that the additive signal aligns with the input itself in a self-supervised manner.
Experiments on multiple datasets show that ACL-SV outperforms the baseline and achieves state-of-the-art performance against recent noise-robust SV systems.


Our experimental code was modified based on [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer).


# Data
The [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) datasets were used for training and test.

The train list should contain the identity and the file path, one line per utterance, as follows:
```
id00000 id00000/youtube_key/12345.wav
id00012 id00012/21Uxsk56VDQ/00001.wav
```
The train list for VoxCeleb2 can be download from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt), and the test lists for VoxCeleb1 can be downloaded from [here](https://mm.kaist.ac.kr/datasets/voxceleb/index.html#testlist). 


For data augmentation, the following script can be used to download and prepare.
```
python3 ./dataprep.py --save_path data --augment
```

We also performed an out-of-domain evaluation using [Nonspeech 100](), [VoxSRC23](), and [VC-Mix]() datasets.

Each dataset must be downloaded in advance for training and testing, and its path must be mapped to the docker environment.

# Environment
Docker image (nvcr.io/nvidia/pytorch:23.07-py3) of Nvidia GPU Cloud was used for conducting our experiments.

Make docker image and activate docker container.
```
./docker/build.sh
./docker/run.sh
```

Note that you need to modify the mapping path before running the 'run.sh' file.

# Training

- on a single GPU
```
python3 ./main.py
```

- on multiple GPUs
```
CUDA_VISIBLE_DEVICES=0,1 python3 ./main.py --distributed
```
Use --distributed flag to enable distributed training.

# Test

```
python3 ./main.py --eval

```
```
CUDA_VISIBLE_DEVICES=0,1 python3 ./main.py --distributed --eval
```


# Citation
Please cite if you make use of the code.

```

@inproceedings{kim2025aclsv,
  title={ADDITIVE CONSISTENCY LEARNING FOR ROBUST SPEAKER VERIFICATION IN NOISY ENVIRONMENTS},
  author={Seung-bin Kim and Chan-yeong Lim and Jungwoo Heo and Hyun-seo Shin and Kyo-Won Koo and Jisoo Son, and Ha-Jin Yu},
  booktitle={},
  year={2025}
}
```