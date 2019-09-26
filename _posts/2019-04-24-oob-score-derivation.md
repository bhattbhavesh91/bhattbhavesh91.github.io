---
title:  "Out-of-bag (OOB) error derivation for Random Forests"
date:   2019-04-24 17:00:00
categories: [oob-score]
tags: [oob-score]

---

The RandomForestClassifier is trained using bootstrap aggregation, where each new tree is fit from a bootstrap sample of the training observations . The out-of-bag (OOB) error is the average error for each  calculated using predictions from the trees that do not contain  in their respective bootstrap sample. This allows the RandomForestClassifier to be fit and validated whilst being trained.

In this video, I example how you end up training only 64% of your training data when OOB option is set true in Random Forest.

## To view the video
* [Click here](https://youtu.be/z-w_W_VJbIY){:target="_blank"}
* Click on the image below

[![Out-of-bag (OOB) score for Ensemble Classifiers in Sklearn](http://img.youtube.com/vi/z-w_W_VJbIY/0.jpg)](http://www.youtube.com/watch?v=z-w_W_VJbIY){:target="_blank"}