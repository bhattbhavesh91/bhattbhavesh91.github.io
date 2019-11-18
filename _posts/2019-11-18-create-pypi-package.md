---
title: "How to upload your python package to PyPi using Twine"
date: 2019-11-18 22:00:00
categories: [pypi-package]
tags: [pypi-package]
description: Small Blog to prepare your Python package for publication & learn how to upload your package to PyPI using Twine.
---


#### Make your code publish-ready
- Refer to the following [blog](https://realpython.com/python-application-layouts/) for understanding how you need to structure your python projects.

#### Create a python package
- In order to get started, I created a small Python package which helps you delete specific websites from Google Chrome history. [link](https://github.com/bhattbhavesh91/delete-chrome-history)

#### Configuring Your Package
- In order for your package to be uploaded to PyPI, you need to provide some basic information about it. This information is typically provided in the form of a setup.py file. This is a good [link](https://github.com/navdeep-G/setup.py) to understand the various components of the setup file.

#### Create a PyPi account
- In order for your package to be uploaded to PyPI, you need to have an PyPi account. Register on the following [link](https://pypi.org/)

#### Publishing to PyPI
- Your package is finally ready to meet the world outside your computer! In this section, you’ll see how to actually upload your package to PyPI.
- To upload your package to PyPI, you’ll use a tool called Twine. You can install Twine using Pip as usual:  
&nbsp;  
```sh
$ pip install twine
```
#### Building Your Package
- Packages on PyPI are not distributed as plain source code. Instead, they are wrapped into distribution packages. The most common formats for distribution packages are source archives and Python wheels.
- A source archive consists of your source code and any supporting files wrapped into one tar file. Similarly, a wheel is essentially a zip archive containing your code. In contrast to the source archive, the wheel includes any extensions ready to use.
- To create a source archive and a wheel for your package, you can run the following command:  
&nbsp;  
```sh
$ python setup.py sdist bdist_wheel
```
* This will create two files in a newly created dist directory, a source archive and a wheel inside the **dist** folder
  * chrome_delete-0.0.8-py3-none-any.whl
  * chrome_delete-0.0.8.tar.gz

#### Upload your package to PyPi
* Twine also checks that your package description will render properly on PyPI. You can run twine check on the files created in dist using the command:  
&nbsp;  
```sh
$ twine check dist/*
```
&nbsp;  
* The final step is here :-
```sh
$ twine upload dist/*
```

### Want to know more about me?
## Follow Me
<a href="https://twitter.com/_bhaveshbhatt" target="_blank"><img class="ai-subscribed-social-icon" src="/assets/images/tw.png" width="30"></a>
<a href="https://www.youtube.com/bhaveshbhatt8791/" target="_blank"><img class="ai-subscribed-social-icon" src="/assets/images/ytb.png" width="30"></a>
<a href="https://github.com/bhattbhavesh91" target="_blank"><img class="ai-subscribed-social-icon" src="/assets/images/gthb.png" width="30"></a>
<a href="https://www.linkedin.com/in/bhattbhavesh91/" target="_blank"><img class="ai-subscribed-social-icon" src="/assets/images/lnkdn.png" width="30"></a>
