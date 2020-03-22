---
title:  "Variance Inflation Factor (VIF) for Detecting Multicolinearity in Python"
date:   2019-05-14 17:00:00
categories: [multicollinearity]
tags: [multicollinearity]

---

Colinearity is the state where two variables are highly correlated and contain similiar information about the variance within a given dataset. To detect colinearity among variables, simply create a correlation matrix and find variables with large absolute values. In R use the corr function and in python this can by accomplished by using numpy's corrcoef function.

Multicolinearity on the other hand is more troublesome to detect because it emerges when three or more variables, which are highly correlated, are included within a model. To make matters worst multicolinearity can emerge even when isolated pairs of variables are not colinear.

The Variance Inflation Factor (VIF) is a measure of colinearity among predictor variables within a multiple regression. It is calculated by taking the the ratio of the variance of all a given model's betas divide by the variane of a single beta if it were fit alone.

## To view the video
* [Click here](https://youtu.be/qmt7ZZoiDwc){:target="_blank"}
* Click on the image below

[![Variance Inflation Factor](http://img.youtube.com/vi/qmt7ZZoiDwc/0.jpg)](http://www.youtube.com/watch?v=qmt7ZZoiDwc){:target="_blank"}

### Want to know more about me?
## Follow Me
<a href="https://twitter.com/_bhaveshbhatt" target="_blank"><img class="ai-subscribed-social-icon" src="/assets/images/tw.png" width="30"></a>
<a href="https://www.youtube.com/bhaveshbhatt8791/" target="_blank"><img class="ai-subscribed-social-icon" src="/assets/images/ytb.png" width="30"></a>
<a href="https://www.youtube.com/PythonTricks/" target="_blank"><img class="ai-subscribed-social-icon" src="/assets/images/python_logo.png" width="30"></a>
<a href="https://github.com/bhattbhavesh91" target="_blank"><img class="ai-subscribed-social-icon" src="/assets/images/gthb.png" width="30"></a>
<a href="https://www.linkedin.com/in/bhattbhavesh91/" target="_blank"><img class="ai-subscribed-social-icon" src="/assets/images/lnkdn.png" width="30"></a>
