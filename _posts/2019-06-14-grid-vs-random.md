---
title:  "Grid vs Random Search Hyperparameter Tuning using Python"
date:   2019-06-14 17:00:00
categories: [hyperparameter-tuning, random-search, grid-search]
tags: [hyperparameter-tuning, random-search, grid-search]

---

In Grid Search, we try every combination of a preset list of values of the hyper-parameters and evaluate the model for each combination. The pattern followed here is similar to the grid, where all the values are placed in the form of a matrix. Each set of parameters is taken into consideration and the accuracy is noted. Once all the combinations are evaluated, the model with the set of parameters which give the top accuracy is considered to be the best.

In Random Search, we try random combinations of the hyperparameters which are used to find the best solution for the built model. It tries random combinations of a range of values. To optimise with random search, the function is evaluated at some number of random configurations in the parameter space. The chances of finding the optimal parameter are comparatively higher in random search because of the random search pattern where the model might end up being trained on the optimised parameters without any aliasing.

In this video, I will focus on two methods for hyperparameter tuning - Grid v/s Random Search and determine which one is better.


## To view the video
* [Click here](https://youtu.be/Ah4wsTXghwI){:target="_blank"}
* Click on the image below

[![Random Search for Hyper-parameter Optimization](http://img.youtube.com/vi/Ah4wsTXghwI/0.jpg)](http://www.youtube.com/watch?v=Ah4wsTXghwI){:target="_blank"}