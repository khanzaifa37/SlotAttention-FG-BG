## ContextFusion and Bootstrap: An Effective Approach to Improve Slot Attention-Based Object-Centric Learning
[📄 arXiv:2509.02032](https://arxiv.org/abs/2509.02032)

---

This repository contains the official implementation of our work, which is an extension of 
[CVPR 2025: Pay Attention to the Foreground in Object-Centric Learning]. 

The main differences from CVPR are as follows:  

1. **Fusion part**: We replace the original feature fusion with slot fusion, i.e., the *ContextFusion* stage.  
2. **Encoder part**: We introduce a feature-adaptive layer after the encoder to mitigate the impact of directly modifying the encoder, i.e., the *Bootstrap Branch*.  


<div align="center">
  <img width="100%" alt=" illustration" src=".github/method.png">
</div>

---




## Contents

- [Installation](#installation)
- [Dataset](#Dataset)
- [Training](#training)
- [Evaluation](#evaluation)



---

## Installation

### Local / Conda

```bash
conda create -n CTFBTP python=3.9.16

conda activate CTFBTP

pip install -r requirements.txt
```

> All experiments run on a single A6000 GPU.

### Google Colab

Colab does not use Conda by default, so use the provided pip-based setup instead.

```bash
git clone https://github.com/khanzaifa37/SlotAttention-FG-BG.git
cd SlotAttention-FG-BG
bash setup_colab.sh
```

Notes for Colab:

- PyTorch is usually preinstalled, so `setup_colab.sh` only installs the extra Python packages needed by this repo.
- `train.py` now supports `--device auto`, which will pick GPU when Colab provides one and fall back to CPU otherwise.
- `train.py` also defaults to `num_workers=0` inside Colab, which avoids common notebook multiprocessing issues.
- If your datasets or checkpoints live in Google Drive, mount Drive first and pass the mounted paths to `--data_path`, `--teacher_checkpoint_path`, and `--log_path`.

Example Colab cells:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!git clone https://github.com/khanzaifa37/SlotAttention-FG-BG.git
%cd /content/SlotAttention-FG-BG
!bash setup_colab.sh
```

---

## Dataset Preparation

### COCO
Download COCO dataset (`2017 Train images`,`2017 Val images`,`2017 Train/Val annotations`) from [here](https://cocodataset.org/#download) and place them following this structure:
```bash
COCO2017
   ├── annotations
   ├── train2017
   └── val2017
```

### PASCAL VOC 2012

Download PASCAL VOC 2012 dataset from `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar`, extract the files and copy `trainaug.txt` in `VOCdevkit/VOC2012/ImageSets/Segmentation`. The final structure should be the following:

```bash
VOCdevkit
   └── VOC2012
          ├── ImageSets
          │      └── Segmentation
          │             ├── trainaug.txt
          │             └── ... 
          │             
          ├── JPEGImages
          ├── SegmentationClass
          └── SegmentationObject
```
### MOVi-C/E
It seems that the commonly used MOVi download site has been closed. We will organize the dataset and upload it later.



### Training(Using the DINOSAUR on the VOC dataset as an example.)
```bash
python train.py --dataset voc --data_path /path/to/VOC2012/ --num_slots 6 --epochs 100 --init_method shared_gaussian --train_permutations standard --teacher_checkpoint_path /path/to/teacher_checkpoint --log_path /path/to/logs --checkpoint_path /path/to/dinosaur_checkpoint
```

Example Colab command:

```bash
python train.py --dataset voc --data_path /content/drive/MyDrive/VOCdevkit/VOC2012 --num_slots 6 --epochs 100 --init_method shared_gaussian --train_permutations standard --teacher_checkpoint_path /content/drive/MyDrive/checkpoints/teacher.pt --log_path /content/drive/MyDrive/contextfusion_logs --checkpoint_path /content/drive/MyDrive/checkpoints/student_resume.pt --device auto
```



### Evaluation(Using the DINOSAUR on the VOC dataset as an example.)

```bash 
python eval.py  voc --data_path /path/to/VOC2012/ --num_slots 6  --teacher_checkpoint_path /path/to/teacher_checkpoint --checkpoint_path /path/to/dinosaur_checkpoint
```



## License

This project is licensed under the MIT License.

## Acknowledgement

This repository is built using the [SPOT](https://github.com/gkakogeorgiou/spot) 

## Citation
If you find this repository useful, please consider giving a star :star: and citation:
```
@inproceedings{tian2025pay,
  title={Pay attention to the foreground in object-centric learning},
  author={Tian, Pinzhuo and Yang, Shengjie and Yu, Hang and Kot, Alex},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={30281--30290},
  year={2025}
}
```
