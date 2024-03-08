# Projet 8 : Réaliser un Dashboard

# Dashboard Interactif 

## Description

Ce projet concerne le développement et le déploiement d'un Dashboard de scoring de crédit pour "Prêt à dépenser", une entreprise financière spécialisée dans l'octroi de crédits à la consommation pour les personnes ayant peu ou pas d'historique de crédit.
L'objectif principal du Dashboard  est de calculer automatiquement la probabilité de remboursement d'un crédit par un client et de classer la demande comme accordée ou refusée en fonction d'un seuil optimisé du point de vue métier.
Le Dashboard fournit une interface intuitive pour interagir avec l'API de scoring de crédit, permettant aux utilisateurs de tester et de visualiser les performances et les résultats du modèle de scoring.

## Architecture du Projet

Le dashboard utilise le framework Streamlit et appelle une API qui repose sur un modèle de scoring développé à partir de données comportementales et financières variées.
Le projet suit une approche MLOps pour l'entraînement, le suivi, et le déploiement du modèle, en utilisant des outils tels que MLFlow pour le tracking des expérimentations, un registre centralisé des modèles, et GitHub Actions pour l'intégration et le déploiement continu.

Dépôt git du  projet API : https://github.com/xhermesxenl/oc_p8_api.git


## Découpage des Dossiers

- `/`: Code source du dashboard, le déploiement du dashboard, liste des packages.


## Installation
Pour installer et exécuter le tableau de bord Streamlit localement, suivez ces étapes :

Clonez ce dépôt.
Installez les dépendances nécessaires :

pip install -r requirements.txt

Exécutez le tableau de bord Streamlit :

streamlit run dashboard/app_streamlit.py

## Déploiement
Les instructions pour le déploiement de l'API et du Dashboard sur une plateforme cloud (Heroku) sont fournies dans le fichier DEPLOYMENT.md

## Utilisation
Après avoir démarré le tableau de bord Streamlit, vous pouvez :

Saisir des données clients pour obtenir des scores de crédit.
Visualiser les résultats et les métriques clés du modèle.

## Fonctionnalités
- Interface intuitive pour tester l'API de scoring de crédit.
- Visualisation des prédictions de scoring et des métriques associées.
- Possibilité de tester différents scénarios clients.

## Licence
[MIT](https://choosealicense.com/licenses/mit/)

## Remarques Supplémentaires
Pour toute question ou suggestion, n'hésitez pas à ouvrir un issue sur ce dépôt.
