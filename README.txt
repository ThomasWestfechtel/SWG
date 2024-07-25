Usage:

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
