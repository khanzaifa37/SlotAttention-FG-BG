## ContextFusion and Bootstrap: An Effective Approach to Improve Slot Attention-Based Object-Centric Learning

---

This repository contains the official implementation of our work, which is an extension of 
[CVPR 2025: Pay Attention to the Foreground in Object-Centric Learning]. 
The extended version has been submitted to the International Journal of Computer Vision (IJCV).


<div align="center">
  <img width="100%" alt=" illustration" src="/home/ysj/ijcv/spot/.github/method.png">
</div>

---

## News
⚠️ **Status**: This repository is **under active development**.  The current release is **not the full version**. More code will be released gradually.



## Contents

- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)



---

## Installation

```bash
conda create -n spot python=3.9.16

conda activate spot

pip install -r requirements.txt
```

> All experiments run on a single A6000 GPU.

---




### Training(Using the DINOSAUR on the VOC dataset as an example.)
```bash
python train.py --dataset voc --data_path /path/to/VOC2012/ --num_slots 6 --epochs 100 --init_method shared_gaussian --train_permutations standard  --teacher_checkpoint_path /path/to/teacher_checkpoint  --log_path /path/to/logs  --checkpoint_path /path/to/dinosaur_checkpoint 
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
