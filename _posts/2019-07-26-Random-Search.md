---
title:  "Random Search for Hyper-parameter Optimization"
date:   2019-07-26 17:00:00
categories: [hyperparameter-tuning, random-search]
tags: [hyperparameter-tuning, random-search]

---

Random search is a technique where random combinations of the hyperparameters are used to find the best solution for the built model. It is similar to grid search, and yet it has proven to yield better results comparatively. 

Instead of searching over the entire grid, random search only evaluates a random sample of points on the grid. This makes random search a lot cheaper than grid search. Random search wasn’t taken very seriously before. This is because it doesn’t search over all the grid points, so it cannot possibly beat the optimum found by grid search. But then along came Bergstra and Bengio. They showed that, in surprisingly many instances, random search performs about as well as grid search. All in all, trying 60 random points sampled from the grid seems to be good enough.

In this video, I'll show you how random search performs about as well as grid search with less number of iterations.

## To view the video
* [Click here](https://youtu.be/wseNcn-Op48)
* Click on the image below

[![Random Search for Hyper-parameter Optimization](http://img.youtube.com/vi/wseNcn-Op48/0.jpg)](http://www.youtube.com/watch?v=wseNcn-Op48)