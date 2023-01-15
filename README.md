### L&T EduTech Hackathon Solution - Yash Khandelwal

This repository is my solution for [Problem Statement 3](https://unstop.com/hackathon/lt-edutech-hackathon-at-shaastra-iitm-shaastra-2023-indian-institute-of-technology-iit-madras-579093). 
![IMG](https://d8it4huxumps7.cloudfront.net/uploads/images/opportunity/banner/63b9583cccb14_lt-edutech-hackathon-at-shaastra-iitm.png?d=1920x557)

## Background

Natural disasters and atmospheric anomalies demand remote monitoring and maintenance of naval objects especially big-size ships. For example, under poor weather conditions, prior knowledge about the ship model 🚢 and type helps the automatic docking system process to be smooth. Hence, a ship or vessel detection system can be devloped  and used in a wide range of applications 📊, in the areas of maritime safety for disaster prevention, fisheries management, marine pollution, defense and maritime security, protection from piracy, illegal migration, etc. In this repository, I showcase the modelling of a Deep Learning model that can be deployed and used in an automated system to identify ship type only from the images taken by the survey boats ✔️

## Data Exploration
 
There are 6,252 images in the training data & 2,680 images in test data.
[Data Source](https://www.kaggle.com/datasets/arpitjain007/game-of-deep-learning-ship-datasets). The images belong to 5 classes, namely:
- 

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Modelling

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Experimental Results

## Inference

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## References

>📋  Pick a licence and describe how to contribute to your code repository.
