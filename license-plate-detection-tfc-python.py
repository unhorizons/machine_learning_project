# %% [markdown]
# ## Import Libaries
# - `keras` comes with `tensorflow`
# - `sklearn`
# - `numpy`
# - `matplotlib`

# %%
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
import cv2
import os
import glob

from random import sample, randint

from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.resnet_v2 import ResNet50V2

import keras.optimizers as  optimizers
import keras.losses as losses

from alpr_lib import AlprTrainer as trainer

import easyocr

from lxml import etree

# %% [markdown]
# ## Chargement du Dataset (Load Dataset)
# Le Dataset contient 
# - Les annotations sont des fichiers xml, qui contient les coordonnées des regions intérêt sur les images (les étiquettes) 
# - Les differentes images des plaques d'immatriculations
# 

# %%
#parcourir les fichiers images et dataset
src_path = "dataset/input"
for dirName, _, fileNames in os.walk('dataset/input'):
    for fileName in fileNames:
        print(os.path.join(dirName, fileName))

# %% [markdown]
# ## Extraction
# Extraire les informations
# - Les images seront stocker dans une liste `X` 
# - Les coordonnées seront extrait des annotations (étiquettes) et stocker dans un vecteur `y = [vx,vy]`
# Note: <i>avec`vx` pour vecteur x et `vy` pour vecteur y, le vecteur x et y contiennent 2 valeur (min et max)</i>
# 
# |  vx  |  vy  |
# |------|------|
# | xmin | ymin |
# | xmax | ymax | 
# 

# %% [markdown]
# #### Extraction d'images

# %%
# Définir taille d'image 
IMAGE_SIZE = 200

img_dir = src_path + "/images" ##"../input/car-plate-detection/images"
data_path = os.path.join(img_dir,'*g') # all files end with ...g (png, jpg, )
files = glob.glob(data_path)
files.sort()

X=[] # List des
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
    X.append(np.array(img))

# %% [markdown]
# #### Extraction des coordonnées (dans les annotations)

# %% [markdown]
# La fonction `resizeannotation` permet d'extraire les cordonnées de la region d'intérêt d'une images
# et les retourne sous dans une liste :  [`ymin`, `ymax`, `xmin`, `xmax`]

# %%
def resizeannotation(f) -> list:  # Fonction qui va extraire les informations des fichiers xml
    tree = etree.parse(f)
    for dim in tree.xpath("size"):
        width = int(dim.xpath("width")[0].text)
        height = int(dim.xpath("height")[0].text)
    for dim in tree.xpath("object/bndbox"):
        xmin = int(dim.xpath("xmin")[0].text) / (width / IMAGE_SIZE)
        ymin = int(dim.xpath("ymin")[0].text) / (height / IMAGE_SIZE)
        xmax = int(dim.xpath("xmax")[0].text) / (width / IMAGE_SIZE)
        ymax = int(dim.xpath("ymax")[0].text) / (height / IMAGE_SIZE)
    return [int(xmax), int(ymax), int(xmin), int(ymin)]

# %%
path = src_path + '/annotations'
text_files = [src_path + '/annotations/'+f for f in sorted(os.listdir(path))] # Stocker les fichiers xml dans la liste

y=[] # List des coordonnées des regions intérêt
for i in text_files:
    y.append(resizeannotation(i))

# %%
#for img in text_files: print(img)

## uncomment to print all images pixel in X

#for img in X: print(img)

# %% [markdown]
# ## Afficher les images du dataset (Display Image Dataset)

# %%
sample_index = []

for i in range(9):
    sample_index.append(randint(0, len(X)))

plt.figure(figsize=(15, 15))
for index, i in enumerate(sample_index):
    plt.subplot(3, 3, index + 1)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB))

# %% [markdown]
# ## Verifier la taille des images

# %% [markdown]
# - ${X}$ : 433 images de tailles 200x200 pixel en couleur dimension 3 <=> (red, green, bleu) 
# - ${y}$ : 433 annotations decrivant chaque region intérêt, dimesion 4 <=> [`ymin`, `ymax`, `xmin`, `xmax`]
# 
# Exemple :
# 
# Le modèle linéaire qu'ont peut en tirer, s'il était question d'une regression linéaire :<br>
# <b>Généraliste</b> :
# - ${ y =Xw + \varepsilon} $ avec ${w}$ le vecteur des paramètres aléatoire attribué par le CNN, qu'il faudra entrainé ensuite. 
# - ${ \hat{y} =X\hat{w}} $ avec $\hat{w}$ le vecteur des paramètres après entrainement du CNN <br>
# <b>Itérative</b> : ${ \hat{y}_{i}=w_{0}+w_{1}x_{i,1}+\ldots + w_{n}x_{i,n}+\varepsilon_{i}}$ 
# 
# Aprés il faudra faire la minimisation de la <i>somme des carrés des écarts ${MSE}$</i> entre les valeurs prédites ${\hat{y}}$  et les valeurs observées ${y}$ par rapport au vecteur paramètre ${w}$ : ${{\hat {\varepsilon }}=y-{\hat {y}}}.$
# <b>Itérative</b> :
# $${\ \mathrm {MSE} = {\hat {\varepsilon }}=\sum _{i=1}^{n}(y_{i}-{\hat {y_{i}}})^{2}}$$
# 
# Le objectif de cette régression est d'établir un modèle y = ƒ(x). Une fois ce modèle estimée, on va chercher à prédire une valeur de `y` pour une valeur de `x` donnée.
# - avec `y` : <i>region intérêt ou plaque d'immatriculation</i> et `x` : <i>images</i>    
# 
# Fin Exemple;
# 
# Note : pour ce projet il s'agit d'une classification à une seule classe. 

# %%
# verificarion la taille des images
print(f"X Shape : {np.array(X).shape}")
print(f"y Shape : {np.array(y).shape}") 

# %% [markdown]
# ## Afficher les images du dataset avec leurs annotations (région d'interêt) 

# %%
plt.figure(figsize=(15, 15))
for index, i in enumerate(sample_index):
    plt.subplot(3, 3, index + 1)
    image = cv2.rectangle(cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB),(y[i][0],y[i][1]),(y[i][2],y[i][3]),(0, 255, 0), 2)
    plt.imshow(image)
    plt.axis("off")

plt.show()

# %%
for i in range(9):
    print(y[i])

# %% [markdown]
# ## Prétraitement des données (Data Preprocessing)

# %%
X = np.array(X)
y = np.array(y)

X = X / 255 # reduire les valeurs des X en valuer comprise entre 0.0 et 1.0 | pour être utilisé on multipliera par 255
y = y / 255 # reduire les valeurs de y 

# %% [markdown]
# <font size="3">Fractionnement des l'ensemble des données : (70% Entrainement, 10% Validation, 20% Testes) | Data Splitting (70% Training, 10% Validation, 20% Testing)</font>

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Test : 20%
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1) # Validation : 10%

# %%


# %% [markdown]
# ## Implémentation du modèle VGG19

# %%
model = Sequential()
model.add(VGG19(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64,  activation="relu"))
model.add(Dense(4,   activation="sigmoid"))

model.layers[-7].trainable = False

model.summary()

# %% [markdown]
# ## Entrainement du modèle VGG19, 
# 
# - Loss fonction : MSE (mean square error)
# - Optimizer : ADAM (a gradiant descent stochastic)
# 

# %%
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.compile(loss=losses.mean_squared_error(), optimizer = optimizers.Adam(), metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32, verbose=1) # training

# %% [markdown]
# ## Evaluation 

# %% [markdown]
# Illustration de la minimisation de l'erreur (loss) par rapport au temps (mesuré en epoch)

# %%
#plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Courbes de l\'entrainement et de validation') #('Training and validation loss curves (VGG19)')
plt.ylabel('perte (loss)')
plt.xlabel('pas (epoch)')
plt.legend(['entrainement', 'validation'], loc='upper right')
plt.show()

# %% [markdown]
# Illustration de l'évolution de la précision (accuracy) du model,

# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Courbes de précision de formation et de validation') # ('Training and validation accuracy curves (VGG19)')
plt.ylabel('précision (accuracy)')
plt.xlabel('pas (epoch)')
plt.legend(['entrainement', 'validation'], loc='upper left')
plt.show()

# %% [markdown]
# ## Teste du VGG19

# %%
test_loss, test_accuracy = model.evaluate(X_test, y_test,steps=int(100))
y_cnn = model.predict(X_test)

print("")
print(f"Loss : {test_loss * 100}%")
print(f"Accuracy : {test_accuracy * 100}%")

# %%
sample_index = []

img_list = []
for i in range(9):
    sample_index.append(randint(0, len(X_test)))

#text = easyocr.Reader(['en'])
plt.figure(figsize=(15, 15))
for index, i in enumerate(sample_index):
    plt.subplot(3, 3, index + 1)
    plt.axis('off')
    ny = np.copy(y_cnn[i])
    ny = ny * 255
    
    rgb_img = cv2.cvtColor(np.copy(X_test[i]).astype('float32'), cv2.COLOR_BGR2RGB)
    image = cv2.rectangle(rgb_img,(int(ny[0]),int(ny[1])),(int(ny[2]),int(ny[3])),(0, 255, 0))
    img_list.append(image)
    plt.imshow(image)
    
plt.show()

# %% [markdown]
# # FIN du code

# %% [markdown]
# 

# %% [markdown]
# 
# ### La suite 
# Juste après aussi implémenter le VGG16 pour voir si ses performances équivalent VGG19.
# 
# Enfin faire l’étude comparative de différent optimiseur : ADAM, ADADELTA, NAG, Momentum, RMSprop,


