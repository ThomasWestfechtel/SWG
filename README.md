# SWG
PyTorch Code for SWG - Combining inherent knowledge of vision-language models with unsupervised domain adaptation through strong-weak guidance
https://arxiv.org/abs/2312.04066

### Method overview
Unsupervised domain adaptation (UDA) tries to overcome the tedious work of labeling data by leveraging a labeled source dataset and transferring its knowledge to a similar but different target dataset. Meanwhile, current vision-language models exhibit remarkable zero-shot prediction capabilities. 
In this work, we combine knowledge gained through UDA with the inherent knowledge of vision-language models.
We introduce a strong-weak guidance learning scheme that employs zero-shot predictions to help align the source and target dataset. For the strong guidance, we expand the source dataset with the most confident samples of the target dataset. Additionally, we employ a knowledge distillation loss as weak guidance.
The strong guidance uses hard labels but is only applied to the most confident predictions from the target dataset. Conversely, the weak guidance is employed to the whole dataset but uses soft labels. The weak guidance is implemented as a knowledge distillation loss with (shifted) zero-shot predictions.
We show that our method complements and benefits from prompt adaptation techniques for vision-language models.
We conduct experiments and ablation studies on three benchmarks (OfficeHome, VisDA, and DomainNet), outperforming state-of-the-art methods. Our ablation studies further demonstrate the contributions of different components of our algorithm.

### Requirements
Beside the requirements.txt this project employs:  
pip install git+https://github.com/openai/CLIP.git  
pip install git+https://github.com/ildoonet/pytorch-randaugment  

### Usage
For ViT backbone:  
To train the network on OfficeHome: 	bash train_oh_ViT.sh 0  
where 0 is the task id [0...12]  
To train the network on VisDA: 	bash train_vis_ViT.sh  

For ResNet backbone:  
To train the network on OfficeHome: 	bash train_oh_res.sh 0  
where 0 is the task id [0...12]  
To train the network on VisDA: 	bash train_vis_res.sh  

You have to change s_dset_path and t_dset_path to the location of the dataset.  

label_c.txt consists of the full path to the image file and the class id  
Example from art of OfficeHome:  

/home/xxx/OfficeHome/Art/Alarm_Clock/00001.jpg 0  
/home/xxx/OfficeHome/Art/Alarm_Clock/00002.jpg 0  
/home/xxx/OfficeHome/Art/Alarm_Clock/00003.jpg 0  
/home/xxx/OfficeHome/Art/Alarm_Clock/00004.jpg 0  
/home/xxx/OfficeHome/Art/Alarm_Clock/00005.jpg 0  
...  
/home/xxx/OfficeHome/Art/Backpack/00001.jpg 1  
/home/xxx/OfficeHome/Art/Backpack/00002.jpg 1  
/home/xxx/OfficeHome/Art/Backpack/00003.jpg 1  
/home/xxx/OfficeHome/Art/Backpack/00004.jpg 1  

### Zero Shot predictions
To get the zero-shot predictions follow the instructions in Predictions folder

### Version 2
For version 2 of the paper the following lines have to be ajusted in the script file:  
perc=('0.5') -> perc=('0.33' '0.66')  
for((h_id=0; h_id < 1; h_id++)) -> for((h_id=0; h_id < 2; h_id++))  
Furthermore, setting the augmentation aug_flac='1' improves the results.  
To change to DAPL predictions set pretext='1' - due to the size only the predictions for OH and VisDA for seed 42 for the ViT backbone are uploaded

### Citation
If you use SWG code please cite:
```text
@article{SKD,
  title={Combining inherent knowledge of vision-language models with unsupervised domain adaptation through strong-weak guidance},
  author={Westfechtel, Thomas and Zhang, Dexuan and Harada, Tatsuya},
  journal={arXiv preprint arXiv:2312.04066},
  year={2023}
}
```
