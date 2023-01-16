### L&T EduTech Hackathon Solution - Yash Khandelwal

This repository is my solution for [Problem Statement 3](https://unstop.com/hackathon/lt-edutech-hackathon-at-shaastra-iitm-shaastra-2023-indian-institute-of-technology-iit-madras-579093). 
![IMG](https://d8it4huxumps7.cloudfront.net/uploads/images/opportunity/banner/63b9583cccb14_lt-edutech-hackathon-at-shaastra-iitm.png?d=1920x557)

## Background

Natural disasters and atmospheric anomalies demand remote monitoring and maintenance of naval objects especially big-size ships. For example, under poor weather conditions, prior knowledge about the ship model 🚢 and type helps the automatic docking system process to be smooth. Hence, a ship or vessel detection system can be devloped  and used in a wide range of applications 📊, in the areas of maritime safety for disaster prevention, fisheries management, marine pollution, defense and maritime security, protection from piracy, illegal migration, etc. In this repository, I showcase the modelling of a Deep Learning model that can be deployed and used in an automated system to identify ship type only from the images taken by the survey boats ✔️

## Data Exploration
[Data Source](https://www.kaggle.com/datasets/arpitjain007/game-of-deep-learning-ship-datasets)

There are 6,252 images in the **training data** & 2,680 images in **test data**. The images belong to 5 classes, namely:
- Cargo
- Military 
- Carrier
- Cruise
- Tankers

Random Sample of images from each class:

![IMG](https://github.com/YashK07/Sol/blob/master/Readme%20Images/S2.png?raw=true)
![IMG](https://github.com/YashK07/Sol/blob/master/Readme%20Images/S3.png?raw=true)



Image count of each classes in the Training Data & after train-validation split:

![IMG](https://github.com/YashK07/Sol/blob/master/Readme%20Images/EDA1.png?raw=true)

- The dataset has a decent balance of class images in the training data which is preserved even after a **train-validation split** (random_state = 42).


## Modelling

Used the technique of **Transfer Learning** to devlope a Deep Learning model for the given task. 

I have used two appraoches for devlopeing models and obtaining the most optimal based on the Cohen Kappa metric:
![IMG](https://github.com/YashK07/PS3-Solution-by-Yash-Khandelwal/blob/master/Readme%20Images/approach.png?raw=true)

In the approach 1, the head has a **Flatten layer** followed by a hidden layer with **512 neurons** & then a predicition layer with 5 neurons (5 classes).

In the appraoch 2, I have used GlobalPooling instead of **Flatten** + **Hidden layers** in the head. Here are a few reasons:

- **Global Pooling** condenses all of the feature maps into a single one, pooling all of the relevant information into a single map that can be easily understood by a single dense classification layer instead of multiple layers.
- It's typically applied as average pooling (GlobalAveragePooling2D) or max pooling (GlobalMaxPooling2D) and can work for 1D and 3D input as well.
- Note that bottleneck layers for networks like ResNets count in tens of thousands of features, not a mere 1536. When flattening, you're torturing your network to learn from oddly-shaped vectors in a very inefficient manner.

Pre-trained Models used in modelling:
- ResNet50
- VGG16
- XCeption 

Above models evaluation on **ImageNet** data: 
| Model	| Size (MB) |	Top-1 Accuracy | Top-5 Accuracy |	Parameters |	Depth	Time (ms) per inference step (CPU) |	Time (ms) per inference step (GPU)
| ------ | --------- | --------------| --------------- | ---------- | ---------------------------------------- | -----------------------------------|
| Xception |	88 |	79.0% |	94.5%	| 22.9M	| 81 | 109.4 |	8.1 |
| VGG16 |	528 |	71.3% |	90.1% |	138.4M |	16 |	69.5 |	4.2  | 
| VGG19 | 549 |	71.3% |	90.0% |	143.7M |	19 |	84.8 |	4.4 |
| ResNet50 |	98 |	74.9% |	92.1% |	25.6M	| 107 |	58.2 |	4.6 | 

I went on with the following models due to their State-of-the-art performance in image classification.

## Experimental Results on Ship DataSet.

### Approach 1
I began with **ResNet50** to obtain a baseline model.
- Augumentations used = rotation, horizontal flip, width shift, height shift + pixel values scaled (0-1)
- Batch size = 64
- Learning rate = 1e^-4
- Optimizer = RMSProp
- Loss = Categorical Cross Entropy
- Metric = Cohen Kappa
- Epochs = 200

Results:
![IMG](https://github.com/YashK07/PS3-Solution-by-Yash-Khandelwal/blob/master/Readme%20Images/approach%201%20resnet50.png?raw=true)
- After 175th epoch, the model gradually starts to overfit. By 200th epoch, it achieves, loss = 0.8804, cohen_kappa = 0.5358, val_loss = 0.9585,  val_cohen_kappa = 0.5346. Total training time for 200 epochs = 5hrs(approx).
- This approach gives suboptimal results. Even, increasing the number of epochs, or playing around with other hyperparamters and using different pre-trained model doesn't seem to work well as this is time consuming & model isn't genralizing well on the validaiton data. Use of flatten layer & hidden layer in the head part increases a lot of **trainable parameters = 51,383,301** making the model more **complex & prone to overfitting**. 


### Approach 2

![IMG](https://github.com/YashK07/PS3-Solution-by-Yash-Khandelwal/blob/master/Readme%20Images/experimental%20results.png?raw=true)

Training for 30 epochs.
Metric Evaluation on Ship Dataset (following are the best results obtained in 30 epoch training)
| Model	| Training Loss |	Training Kappa | Training Accuracy |	Validation Loss |	Validation Kappa |	Validation Accuracy
| ------ | ------------ | --------------| ------------------ | ---------- | ---------------------- | ---------------------|
| Xception |	0.0746 |	0.9608 |	0.9697	| 0.3965	| 0.8963 | 0.9200 |
| VGG16 |	0.3257 |	0.8528 |	0.8867 |	0.3632 |	0.8375 |	0.8731 |	 
| ResNet50 |	0.1016 |	0.9532 |	0.9639 |	0.6208	| 0.8704 |	0.8998 |	

Obtained the highest **Kappa Score** on validation data with **Xception**. Moreover, all other metrics and loss values are most optimal in the case of Xception. Hence, I chose Xception as the **baseline**.

After multiple experiments, with the following set of **hyperparamters** on Xception I obtained the final model:
- Augumentations used = rotation, horizontal flip, width shift, height shift, zoom, vertical flip + pixel values scaled (0-1)
- Batch size = 64
- Learning rate = 3e^-4
- Optimizer = Adam
- Loss = Categorical Cross Entropy
- Metric = Cohen Kappa, F1 Score, Categorical Accuracy
- Epochs = 50

Results:
| Model	| Training Loss |	Training Kappa | Training F1 |	Training Accuracy | Validation Loss |	Validation Kappa | Validation F1 |	Validation Accuracy |
| ------ | ------------ | ---------------| ------------| ------------------- | -------------- | -----------------|---------------|----------------------|
| Xception |	0.0956 |	0.9559 |	0.9674	| 0.9659	| 0.2920 | 0.9081 | 0.9357 | 0.9286 |

## Inference

The prediction **submission file** is present in [PS3 Deep Learning Solution](https://github.com/YashK07/PS3-Solution-by-Yash-Khandelwal/tree/master/PS3%20Deep%20Learning%20%20Solution) directory along with training and inference python notebook. Download the **final trained model** from [here](https://drive.google.com/file/d/1vNPNdv3m436L07kFrmnxQVDnuP7_cU4T/view?usp=share_link). 

## Conclusion
