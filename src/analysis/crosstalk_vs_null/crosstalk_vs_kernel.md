# Crosstalk et kernel-null

## But

Ces analyses quantifient la dégradation d'un nuller à noyau N4x4-T8 lorsque les
matrices d'injection et de sortie contiennent du crosstalk. La première analyse
(`crosstalk_vs_null.py`) mesure un null classique par comparaison entre la sortie
brillante et la sortie annulée. La seconde (`crosstalk_vs_kernel.py`) étudie une
quantité directement liée au kernel-null: la différence absolue entre les deux
sorties sombres en quadrature de phase, `|Dark 1 - Dark 2|`.

## Modèle

Pour chaque niveau de crosstalk, deux matrices unitaires complexes sont tirées
aléatoirement. Elles sont obtenues comme exponentielles de matrices
anti-hermitiennes, puis ajustées par dichotomie afin que le plus grand terme
hors diagonale ait le niveau demandé. Leur unitarité assure la conservation de
l'énergie optique.

Le contexte PHOB utilisé pour chaque tirage est monochromatique, possède une
caméra idéale, des phases injectées nulles et un OPD statique nul dans le chip.
Les seules imperfections permanentes sont donc les deux matrices de crosstalk.

## Grille 2D

Pour chaque couple `(crosstalk, RMS de cophasage)`, plusieurs vecteurs de quatre
pistons d'entrée sont tirés selon une loi normale centrée. L'écart-type de cette
loi est le RMS demandé en nanomètres. Chaque observation est propagée dans le
contexte, puis le kernel instantané `|Dark 1 - Dark 2|` est calculé. La valeur
d'une réalisation bootstrap est la moyenne de ces observations; la couleur de
la première carte est ensuite la moyenne de ces réalisations bootstrap.

La seconde carte représente l'incertitude relative en chaque point:

$$
\epsilon_\mathrm{rel} =
\frac{\operatorname{std}(K_1,\ldots,K_N)}
     {\operatorname{mean}(K_1,\ldots,K_N)}.
$$

Elle mesure la dispersion due aux réalisations aléatoires de crosstalk, après
avoir moyenné les observations de phase de chaque réalisation.

## Échelles et interprétation

L'axe du crosstalk et l'axe vertical du RMS sont logarithmiques afin de rendre
visibles les régimes faibles et forts. La grille utilisée par le script va de
`10^-3 nm` à `100 nm`. La colorbar du kernel moyen reste logarithmique, tandis
que celle de l'erreur relative est linéaire et exprimée en pourcentage.

## Comparaison avec le null classique

Le script produit également deux cartes analogues pour le null classique. À
chaque observation, celui-ci est défini par le rapport entre la sortie annulée
et la sortie brillante:

$$
N_\mathrm{classique} = \frac{I_\mathrm{Null}}{I_\mathrm{Bright}}.
$$

Les valeurs `Null / Bright` sont moyennées sur les observations de cophasage,
puis sur les réalisations bootstrap. La carte d'incertitude utilise le même
coefficient de variation bootstrap que pour le kernel. Les deux métriques sont
calculées dans les mêmes propagations, avec les mêmes matrices et les mêmes
tirages de pistons, afin de permettre une comparaison directe.

Les points de grille sont calculés en parallèle par processus. Chaque processus
reçoit un point complet et calcule ses réalisations bootstrap, construit ses
propres contextes PHOB et utilise des générateurs aléatoires indépendants
dérivés de la seed globale. Cela évite les accès concurrents à un contexte
partagé, contourne le GIL Python et conserve la reproductibilité. La barre
`tqdm` avance à chaque point de grille terminé, quel que soit son ordre d'arrivée.

Les nombres de bootstrap et d'observations sont indépendants: augmenter le
nombre d'observations réduit le bruit de Monte-Carlo de la moyenne de phase,
tandis qu'augmenter le bootstrap caractérise mieux la dispersion entre
matrices de crosstalk.