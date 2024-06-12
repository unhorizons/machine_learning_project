# Automatic license plate recognition using stochastic gradient descent
## (La reconnaissance des plaques d'immatriculation pour la descente de gradient stochastique)


C'est projet une étude comparatif des différents [optimizers]() axés sur la [descente de gradient stochastique](https://scikit-learn.org/stable/modules/sgd.html) qui permet d’ajuster le classificateur d’images sous une fonction coût convexe,  

## Optimizers

Les optimizers a comparé: 

- <b>SGD : Stochastic Gradient Descent </b> - a variant of the Gradient Descent algorithm used for optimizing machine learning models.
- <b>Momentum</b> - an optimization algorithm that helps accelerate gradient vectors in the right directions, thus leading to faster converging.
- <b>NAG : Nesterov Accelerated Gradient</b> - a way to give our momentum term a “look ahead” feature.
- <b>AdaGrad</b> - an algorithm for gradient-based optimization that adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters.
- <b>RMSprop : Root Mean Square Propagation </b>  - an unpublished, adaptive learning rate method proposed by Geoff Hinton.
- <b>Adadelta </b>  - an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate.
- <b>Adam : Adaptive Moment Estimation</b> - an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iteratively based on training data.
- <b>AdaMax</b>  - a variant of Adam based on the infinity norm.

This paper ([here](https://arxiv.org/pdf/1609.04747)) explains more about optimizers.

## Le Model CNN 

Le Model CNN utilisé ici c’est le VGG16 (et VGG16).

VGG signifie Visual Geometry Group. Il s'agit d'une architecture standard de réseau de neurones à convolution profonde (CNN) à plusieurs couches. Le "profond" fait référence au nombre de couches avec VGG-16 ou VGG-19 consistant respectivement en 16 et 19 couches convolutionnelles. L'architecture VGG est à la base de modèles de reconnaissance d'objets révolutionnaires (this paper [here](https://www.researchgate.net/profile/Srikanth-Tammina/publication/337105858_Transfer_learning_using_VGG-16_with_Deep_Convolutional_Neural_Network_for_Classifying_Images/links/5dc94c3ca6fdcc57503e6ad9/Transfer-learning-using-VGG-16-with-Deep-Convolutional-Neural-Network-for-Classifying-Images.pdf?_sg%5B0%5D=started_experiment_milestone&origin=journalDetail&_rtd=e30%3D) talk more about).

## Structure du projet

- Dans le répertoire principale il y a un fichier jupyter notebook license-plate-detection-tfc1.ipynb qui contient les expérimentations de l'entraînement du modèle VGG16 avec l’optimizer Adam.

- Dans le répertoire optimiser_tested il y a les différents jupyter notebook par optimizer.

- Dans sample il y a un tuto sur le création d’un neurone utilisant la descente de gradient. Avec un exemple sur la classification d’images (faire la différence entre un chien et chat sur une image)

Ce projet ne fait pas du OCR (Optical Character Recognition), la reconnaissance de caractère, le modèle n’est pas entraîné pour ça, mais si vous voulez pousser l'expérience plus loin renseignez vous sur OCR python (voir [EasyOCR](https://www.jaided.ai/easyocr/tutorial/), [Tesseract](https://pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/)) 

## Math 

Ce projet demande la maîtrise de certain notions mathématique et statistiques (pas obligatoire) :

- Matrices et Vecteurs (une image peut être représenté dans une matrices)
- Convolution d’une matrice
-  Convexité
- Dérivée
- Gradient, Hessien, Jacobien
- Régression Linéaire (facultatif: régression logistique)
- Classifiction Linéaire (pas trop mathématique)
- Erreur quadratique moyenne
- Moindres carrés linéaires (facultatif : moindres carrés ordinaires)
- Descente de gradient
- Optimisation stochastique (pour les notions trés avancées)
- Etc.


## Install python lib
before install, update pip : `pip install --upgrade pip`

- tensorflow : `pip install tensorflow`
- opencv : `pip install opencv-python`
- sklearn
- numpy
- matplotlib
- lxml


if you're using Anaconda check [this](https://www.tensorflow.org/install/pip), explain how to install tensorflow on it.

<br>
<br>
<br>

### Note : <i>Only the main `license-plate-detection-tfc1.py` file which is fully commented. Understanding of other files can be inferred from this one</i>.


A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for ml_project_
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes ml_project_ a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```





