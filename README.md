# project_1A

# Projet d'informatique : Réseaux de neuronnes.
Par Edwin Roussin et Alexandre Partensky.



## Gestion de nos données

 La base de données MNIST est un ensemble d'image de chiffre de résolution 28x28, chacune accompagné de leur valeur représentée sous forme d'un entier ou d'un vecteur. Après avoir téléchargé la base de données, nous l'ouvrons à l'aide d'un code que nous avons repris sur Github.

> Lien base MNIST : http://yann.lecun.com/exdb/mnist/



> Lien github : https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_loader.py



  Les images sont triées selon deux groupes. Le premier, appelé "training_data" permet d'entrainer l'algorithme de machine learning à reconnaître les images. Le second, "test_data" permet de tester la capacité de l'algorithme à reconnaître des images, une fois que celui-ci à été entrainé sur les images d'entraînements. Ces deux objets sont importés sous forme de liste de couple, chaque couple représentant une image avec l'ensemble des valeurs des pixels pour la première composante et la traduction de la valeur de l'image pour la seconde composante. Pour le test_data, la valeur de chaque image (deuxième composante du tuple) est un nombre tandis que c'est une liste sous la forme $np.array([0,1,0,0,0,0,0,0,0,0])$ pour le trainig_data. La fonction "vectorized_result" permet de passer de la première représentation à la deuxième si besoin.
  
Il existe également un validation_data utilisé de manière intermédiaire dans la phase d'entrainement qui comporte moins de données que le test_data. Il est surtout utile pour calibrer les hyperparamètres du réseau, prévenir l'over-fitting et mettre en place des early-stopping par exemple.





## Description du type de neuronne utilisé

Le neuronne utilisé est classique.

Pour une liste d'entrée  $(e_{1},e_{2},e_{3},...,e_{n}) $, Il définit en entier $z$ tel que 

$z = w_{1}×e_{1} + w_{2}×e_{2} + ... + w_{n}×e_{n} + b $

avec $(w_{1},w_{2},w_{3},...,w_{n})$ l'ensemble des "poids" associé à un neuronne et $b$ le biais du neuronne. 

On compose ensuite z par la fonction d'activation sigmoid définie ci-dessus. Le neuronne renvoie alors sigmoid(z). 

Pour une couche de neuronnes, prenant en entrée $E = (e_{1},e_{2},e_{3},...,e_{n})$, on peut calculer la valeur de sortie sous forme d'un vecteur $A$ tel que 

$A = sigmoid(Z) = W·E + B $

avec $A$ est le vecteur d'activation de la couche en question, $W$ la matrice des poids des neuronnes de la couche, dont chaque ligne est l'ensemble des poids d'un seul neuronne et $B$ le vecteur des biais des neuronnes de la couche

$A = sigmoid(Z)$ est le vecteur de sortie de la couche de neuronne dont les composantes sont $(sigmoid(z_{1}),sigmoid(z_{2}),...,sigmoid(z_{n})$.

### Description des attributs :

La classe contient 6 attributs :

- **num_layers** est le nombre de couche de neuronnes de notre réseau
>**Par soucis de simplicité, nous décidons d'utiliser uniquement un réseau de neuronnes avec 3 couches, une pour les entrées, une au milieu et une pour les sorties**.
- **sizes** est un tableau de taille num_layers et dont la i-ème case est le nombre de neuronnes dans la i-ème couche du réseau.

- **biaises** est la liste des vecteurs des biais de chaque couche de neuronnes.

- **weights** est la liste des matrices des poids de chaque couche de neuronnes.

- **train_cost** est la liste des costs sur le train set lors de l'entraînements.

- **test_cost** est la liste des costs sur le test set lors de l'entraînement
- **acc** la liste donnant l'accuracy du réseau sur le test set au fur et à mesure de l'entraînement


### Description des premières méthodes :

- **init** correspond à l'initialisation de la classe, elle prend en argument la taille de réseaux, soit une liste de longueur variable dont la valeur de la i-ième case est le nombre de neuronne de la i-ème couche.

 Elle prend également en argument, si spécifié, une classe Cost. Cela permet de tester différentes fonctions coût. Nous avons par exemple rajouté la fonction log_loss à notre réseaux. Cela permet d'accéler l'apprentissage surtout au début si le score est très mauvais.

>La première couche, comporte $28*28 = 784$ entrées, soit une par pixel.

>La dernière couche, comporte 10 sorties, la valeur de la i-ème sortie est la probabilité que le chiffre soit égal à i, elle est donc comprise entre 0 et 1 (la fonction sigmoïd est bien à valeur dans $[0;1]$). 
Pour intépreter ce résultat, on considère que le chiffre choisis par l'algorithme est celui dont la probabilité est la plus élévée. 

>Comme nous décidons d'utiliser un réseau avec seulement 3 couches, on n'utilise $init$ uniquement avec un tableau de taille 3.


- **vectorized_result** transforme un chiffre en une représentation sous forme de vecteur qui est semblable à celle de sortie du réseau. Le chiffre i est renvoyé sous la forme d'un vecteur de taille 10 dont toutes les composantes sont nulles sauf la i-ème qui vaut 1. 

- **feedforward** prend en argument la liste des pixels d'une image et renvoie à l'aide d'une suite de calculs matriciels le vecteur de sortie de notre réseau. 

- **accuracy** prend en argument la liste d'image test_data et renvoie le taux de succes de l'algorithme de reconnaissance ainsi que le cost total.



- **progression**  affiche avec des graphes l'erreur cumulée et la précision du réseau de neuronnes.

>On définit une nouvelle classe à chaque fois qu'on souhaite utiliser une nouvelle fonciton coût. Ici il y a le classique QuadraticCost ainsi que la CrossEntropyCost. Pour chaque coût on implémente la fonction coût en elle même ainsi que la fonction delta qui dépend mathématiquement de la dérivée de la fonction coût. 

>La différence majeure entre les deux est que le delta pour la fonction log loss ne dépend plus de $sigma$_$prime(z)$ qui peut être très faible alors que le reseaux se trompent complétement. Ainsi on est assuré que delta ne sera jamais trop petit lorsque l'output du reseau est très différent de la réponse attendu, ce qui n'était pas le cas avec le QuadraticCost.


- **SGD** est la méthode qui excécute l'apprentissage de notre réseau de neurones.

Elle prend donc "training_data" en premier argument. 
"nb_training" est le nombre de fois que le programme répétera l'apprentissage sur l'ensemble du "training_data".

Lors de l'apprentisssage celui-ci est divisé un nombre de lots (batch) de taille batch_size (troisième argument). Le reseau de neurone apprendra ainsi batch après batch.

"eta" est le learning rate, un des hyperparamètres de notre réseau, qui comme son nom l'indique peut de manière analogique est  vue comme une vitesse d'apprentissage du  réseau.

Enfin "test_data" permet entre chaque entraînement sur le "training_data" d'observer l'évolution des performances de l'algorithme sur des données en dehors du set d'apprentissage.

En particulier, SGD divise le l'ensemble des images de training_data en différents lots. Lorsque l'algorithme finit de parcourir un lot, SGD appelle la méthode **update_batch** qui va modifier les weights et les biaises du réseau selon la méthode de descente de gradient obtenue par la méthode **backprop**. 

Séparer en différents lots permet d'éviter que l'algorithme apprenne à chaque image et donc que le temps d'exécution du programme ne soit trop long.

- **update_batch** est une méthode appelée par SGD, qui permet l'apprentissage du réseau lorsqu'on lui fait parcourir un lot d'image. Elle prend donc en argument un batch, et eta, la vitesse d'apprentissage. Pour chaque élément du batch, elle récolte dw et db, soit les modifications du réseau nécéssaire, obtenue par la méthode de descente de gradient et calculé ici par la méthode **backprop**. Elle en fait une moyenne et modifie adéquatement les valeurs de weights et biaises. Plus eta est élevé, plus la modification sera importante.

- **backprop** est une méthode qui prend en argument une image, donc ici la liste de valeur de ses pixels et le nombre qu'elle représente.*La méthode calcule d'abord l'output du réseau pour cette image qui est un vecteur numpy de taille 10 avec des nombre dans $[0,1]$.

 Elle calcule ensuite, selon la méthode de descente du gradient, dw et db, qui sont les modifications qu'il faut apporter au réseaux de neuronnes afin de faire baisser le coût de manière optimale.
 
 La méthode utilisé s'appelle backpropagation. L'idéee générale est de d'abord calculer les dérivées partielles de C (fonction coût quelconque) par rapport au biais et poids de la dernière couche, puis grâce à ces valeurs de calculer les valeurs de la couche précendete etc jusqu'à tous les connaîtres.

Voici les équations en question :

![back_propagtion](https://user-images.githubusercontent.com/74186183/196455899-298b42c3-d9a8-4ac6-bad9-46895d1d07e9.png)



Voici les résultats que ce code permet d'obtenir : 

![result_MNIST](https://user-images.githubusercontent.com/74186183/196456419-cc9e57c9-8004-4549-bbcc-1c516e022fe7.png)
