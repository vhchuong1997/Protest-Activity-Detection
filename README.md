# Protest Activity Detection and Perceived Violence Estimation from Social Media Images
This github covers a range of experiments of the task in the paper **_Protest Activity Detection and Perceived Violence Estimation from Social Media Images_** (ACM Multimedia 2017) [\[arxiv\]](https://arxiv.org/abs/1709.06204) by [Donghyeon Won](dhwon.com), [Zachary C. Steinert-Threlkeld](https://zacharyst.com/), [Jungseock Joo](http://home.jsjoo.com/).

![](https://raw.githubusercontent.com/wondonghyeon/protest-detection-violence-estimation/master/files/overview.png)

### Requirements   
[Pytorch](http://pytorch.org/)   
[NumPy](http://www.numpy.org/)   
[pandas](https://pandas.pydata.org/)   
[scikit-learn](http://scikit-learn.org/)   

### Usage   
#### Training  
To train our different models, please access file ```train.ipynb``` to run the bash codes inside

#### Evaluation
To obtain evaluation of our different models, please access file ```model-performance.ipynb``` to run the bash codes inside

### UCLA Protest Image Dataset   
![](https://raw.githubusercontent.com/wondonghyeon/protest-detection-violence-estimation/master/files/1-d.png)
You will need to download our UCLA Protest Image Dataset to train the model. Please e-mail the author at won.donghyeon@gmail.com  if you want to download the dataset!

#### Dataset Statistics   
\# of images: 40,764   
\# of protest images: 11,659   
##### Protest \& Visual Attributes   

|Fields       |Protest|Sign  |Photo|Fire |Police|Children|Group>20|Group>100|Flag |Night|Shouting|
|-------------|-------|------|-----|-----|--------|--------|--------|---------|-----|-----|-----|
|\# of Images |11,659 |9,669 |428  |667  |792     |347     |8,510   |2,939    |970  |987  |548  |
|Positive Rate|0.286  |0.829 |0.037|0.057|0.068   |0.030   |0.730   |0.252    |0.083|0.085|0.047|
##### Violence   

|Mean |Median |STD  |
|-----|-------|-----|
|0.365|0.352  |0.144|

![](https://raw.githubusercontent.com/wondonghyeon/protest-detection-violence-estimation/master/files/violence_hist.png)

### Model
#### Architectures 
We fine-tuned ImageNet pretrained [EfficientNet](https://arxiv.org/abs/1905.11946) to our data. 
You can download the weights from this [Google Drive link](https://www.dropbox.com/s/rxslj6x01otf62i/model_best.pth.tar?dl=0).



#### Performance

<!-- |Fields  |Protest|Sign  |Photo|Fire |Law Enf.|Children|Group>20|Group>100|Flag |Night|Shout|
|--------|-------|------|-----|-----|--------|--------|--------|---------|-----|-----|-----|
|Accuracy|0.919  |0.890 |0.967|0.980|0.953   |0.970   |0.793   |0.803    |0.921|0.939|0.952|
|ROC AUC |0.970  |0.922 |0.811|0.985|0.939   |0.827   |0.818   |0.839    |0.828|0.940|0.849| -->

|Protest|Sign  |Photo|
|-------|------|-----|
|![][protest-roc]|![][sign-roc]|![][photo-roc]|

|Fire|Police|Children|
|-------|------|-----|
|![][fire-roc]|![][police-roc]|![][children-roc]|

|Group>20|Group>100|Flag|
|-------|------|-----|
|![][group_20-roc]|![][group_100-roc]|![][flag-roc]|

|Night|Shouting|Violence|
|-------|------|-----|
|![][night-roc]|![][shouting-roc]|![][violence-scatter]|

[protest-roc]: https://raw.githubusercontent.com/vhchuong1997/Protest-Activity-Detection/master/files/protest_EffNetB1_2_drop25_adam_0.0001.png
[sign-roc]: https://raw.githubusercontent.com/vhchuong1997/Protest-Activity-Detection/tree/master/files/sign_EffNetB1_2_drop25_adam_0.0001.png?raw=true
[photo-roc]: https://raw.githubusercontent.com/vhchuong1997/Protest-Activity-Detection/tree/master/files/photo_EffNetB1_2_drop25_adam_0.0001.png?raw=true
[fire-roc]: https://raw.githubusercontent.com/vhchuong1997/Protest-Activity-Detection/tree/master/files/fire_EffNetB1_2_drop25_adam_0.0001.png?raw=true
[police-roc]: https://raw.githubusercontent.com/vhchuong1997/Protest-Activity-Detection/tree/master/files/police_EffNetB1_2_drop25_adam_0.0001.png?raw=true
[children-roc]: https://raw.githubusercontent.com/vhchuong1997/Protest-Activity-Detection/tree/master/files/children_EffNetB1_2_drop25_adam_0.0001.png?raw=true
[group_20-roc]: https://raw.githubusercontent.com/vhchuong1997/Protest-Activity-Detection/tree/master/files/group_20_EffNetB1_2_drop25_adam_0.0001.png?raw=true
[group_100-roc]: https://raw.githubusercontent.com/vhchuong1997/Protest-Activity-Detection/tree/master/files/group_100_EffNetB1_2_drop25_adam_0.0001.png?raw=true
[flag-roc]: https://raw.githubusercontent.com/vhchuong1997/Protest-Activity-Detection/tree/master/files/flag_EffNetB1_2_drop25_adam_0.0001.png?raw=true
[night-roc]: https://raw.githubusercontent.com/vhchuong1997/Protest-Activity-Detection/tree/master/files/night_EffNetB1_2_drop25_adam_0.0001.png?raw=true
[shouting-roc]: https://raw.githubusercontent.com/vhchuong1997/Protest-Activity-Detection/tree/master/files/shouting_EffNetB1_2_drop25_adam_0.0001.png?raw=true
[violence-scatter]: https://raw.githubusercontent.com/vhchuong1997/Protest-Activity-Detection/tree/master/files/violence_EffNetB1_2_drop25_adam_0.0001.png?raw=true
