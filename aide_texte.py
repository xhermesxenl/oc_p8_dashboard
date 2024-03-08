
def col_desc() :
    columns_mapping = {
        "SK_ID_CURR": "ID Client",
        "NAME_CONTRACT_TYPE": "Type de Contrat",
        "CODE_GENDER": "Genre",
        "CNT_CHILDREN": "Nombre d'Enfants",
        "AMT_INCOME_TOTAL": "Revenu Total",
        "DAYS_BIRTH": "Âge (jours)",
        "DAYS_EMPLOYED": "Emploi (jours)",
        "EXT_SOURCE_3": "Source Externe 3",
        "EXT_SOURCE_2": "Source Externe 2",
        "AMT_CREDIT": "Montant du Crédit"
    }

    columns_to_display = [
        "ID Client", "Type de Contrat", "Genre",
        "Nombre d'Enfants", "Revenu Total", "Âge (jours)", "Emploi (jours)"
    ]
    return columns_to_display, columns_mapping

#################################################
###         Aide Comparaison Client           ###
#################################################
def aide_comparaison_impact() :
    explication_text = """
        <style>
        .custom-info {
            font-size: 18px;  
            color: #000000; 
            background-color: #e8f2fc; 
            padding: 10px; 
            border-radius: 5px; 
        }
        </style>
        <div class="custom-info">
        <strong>Comparaison des Profils Clients</strong><br>
        Cette visualisation offre une comparaison entre les profils de différents clients en fonction de diverses caractéristiques et de leur statut d'approbation de prêt. Les boîtes à moustaches montrent la distribution des valeurs pour chaque caractéristique, séparées par les catégories d'approbation, permettant d'identifier rapidement les tendances, les écarts et les anomalies.
        <br><br>
        En juxtaposant les données d'un client spécifique à ces distributions, nous pouvons observer où il se situe par rapport aux autres, mettant en évidence les attributs qui peuvent influencer positivement ou négativement la décision de prêt. Le marqueur distinctif du client, coloré selon son statut d'approbation, illustre sa position relative, offrant des insights sur comment ses caractéristiques se comparent à celles des demandeurs réussis ou non.
        <br><br>
        Cette approche permet d'évaluer de manière intuitive l'adéquation d'un client avec les critères d'approbation et d'identifier des domaines potentiels d'amélioration ou de risque. C'est un outil précieux pour comprendre les dynamiques de l'ensemble des données et pour prendre des décisions éclairées basées sur une analyse comparative approfondie.
        </div>
    """
    return explication_text

def aide_comparaison_bi(a,b) :
    explication_text = f"""
                <style>
                .custom-info {{
                    font-size: 18px;  
                    color: #000000; 
                    background-color: #e8f2fc; 
                    padding: 10px; 
                    border-radius: 5px; 
                }}
                </style>
                <div class="custom-info">
                <strong>Explication de l'Analyse Bivariée</strong><br>
                Cette analyse explore la relation entre <strong>{a}</strong> et <strong>{b}</strong>. En comparant ces deux caractéristiques, nous cherchons à identifier des tendances, des corrélations ou des modèles significatifs qui peuvent exister dans notre ensemble de données. Chaque point sur le graphique représente une observation, permettant de visualiser comment les valeurs de <strong>{a}</strong> se rapportent à celles de <strong>{b}</strong>. Cette approche aide à mieux comprendre les dynamiques au sein de notre population étudiée et à découvrir des tendances potentiellement utiles pour des analyses plus approfondies ou des prises de décision éclairées.
                </div>
            """

    return explication_text

#####################################################
###         Aide Feature Global - Local           ###
#####################################################
def afficher_infos_local():
    explication_text = f"""
                 <style>
                .custom-info-global {{
                font-size: 18px;
                color: #000000; 
                background-color: #e8f2fc; 
                padding: 10px;
                border-radius: 5px;
                }}
                </style>
                <div class="custom-info-global">
                    <strong>Explication du graphique en cascade SHAP</strong><br>
                    Le graphique en cascade ci-dessus débute par la valeur attendue de la sortie du modèle, c'est-à-dire la moyenne des prédictions pour l'ensemble des observations. Cette valeur de base est le point de départ pour comprendre l'influence de chaque caractéristique sur la prédiction finale. Chaque barre du graphique, qu'elle soit orientée vers le haut (en rouge) pour une influence positive, ou vers le bas (en bleu) pour une influence négative, représente la contribution spécifique de chaque caractéristique à l'écart entre la valeur attendue et la prédiction finale pour le dossier spécifique du client. Cette visualisation permet de décomposer de manière intuitive comment chaque facteur contribue à la décision finale du modèle, en fournissant une compréhension approfondie de la prédiction à un niveau individuel.
                 </div>
                """
    return explication_text

def afficher_infos_global():
    explication_text = f"""
                 <style>
                .custom-info-global {{
                    font-size: 18px;
                color: #000000; 
                background-color: #e8f2fc; 
                padding: 10px;
                border-radius: 5px;
                }}
                </style>
                <div class="custom-info-global">
                <strong>Compréhension de l'importance globale des caractéristiques</strong><br>
                Ce graphique illustre l'importance globale des différentes caractéristiques prises en compte par le modèle pour effectuer ses prédictions. Chaque barre représente une caractéristique différente et son poids relatif dans la décision du modèle, permettant d'identifier les facteurs ayant le plus fort impact sur les prédictions, indépendamment des cas individuels. Cette vue d'ensemble aide à comprendre quelles caractéristiques le modèle considère comme les plus déterminantes pour évaluer le risque, offrant ainsi une vision claire de la logique suivie par le modèle dans son ensemble.
                </div>
                """
    return explication_text

##############################################
###         Aide Describe Client           ###
##############################################
def afficher_infos_client():
    explication_text = f"""
        <div style="background-color:#e8f2fc; padding:10px; border-radius:10px">
        <h2>Informations Client</h2>
    
        <p><strong>Id Client</strong>: Identifiant unique du client. Sert de clé primaire pour identifier les enregistrements de clients de manière unique.</p>
    
        <p><strong>Type de Contrat</strong>: Type de contrat de crédit. Indique le type de prêt contracté par le client, par exemple, prêt à la consommation ou prêt immobilier.</p>
    
        <p><strong>Genre</strong>: Genre du client. Le sexe du client, généralement indiqué par 'M' pour masculin et 'F' pour féminin.</p>
    
        <p><strong>Nombre d'Enfants</strong>: Nombre d'enfants. Le nombre total d'enfants que le client a.</p>
    
        <p><strong>Revenu Total , AMT_INCOME_TOTAL</strong>: Revenu total annuel. Le revenu annuel total du client.</p>
    
        <p><strong>Âge (année)</strong>: Âge du client (année). L'âge du client, calculé en jours à partir de la date de naissance jusqu'à la date de la demande de prêt.</p>
    
        <p><strong>Emploi (jours)</strong>: Durée de l'emploi (en jours). Le nombre de jours depuis le début de l'emploi actuel du client.</p>
    
        <p><strong>EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3</strong>: Ces trois variables représentent des scores provenant de sources extérieures. Les scores sont typiquement issus d'agences de crédit ou d'autres sources de données externes évaluant la solvabilité du client. Chaque source pourrait avoir sa propre méthodologie de calcul, mais en général, ces scores tentent de prédire la probabilité de défaut de paiement du client. Un score plus élevé indique généralement un risque de défaut plus faible.</p>
    
        <p><strong>AMT_CREDIT</strong>: Montant du crédit. Il s'agit du montant total du prêt que le client demande. Cette valeur représente la somme d'argent que le client souhaite emprunter.</p>
    
        </div>
        """

    return  explication_text

##############################################
###         Aide Menu                      ###
##############################################
def afficher_menu_client():


    explication_text = f"""
        <br />
        <br />
        <div style="background-color:#fffce7; padding:10px; border-radius:10px">
            <h5>
            Veuillez saisir un numéro de client dans la barre latérale à gauche et appuyer sur 'Obtenir la prédiction' pour commencer.</h5>
        </div>
        <br />
        <hr />
        <br />
        <div style="background-color:#e8f2fc; padding:10px; border-radius:10px">
        <h4>Les fonctionnalités du menu</h4>
    
        <p><strong>Informations descriptives du client</strong>: Affiche les informations de base du client telles que le nom, l'âge, et le genre. </p>
    
        <p><strong>Facteurs d'influence locaux</strong>: Montre comment les caractéristiques individuelles du client influencent la prédiction. </p>
        <p><strong>Facteurs d'influence globaux</strong>: Révèle les tendances générales des caractéristiques qui affectent les décisions de prêt.</p>
    
        <p><strong>Comparaison aves l'ensemble des Clients</strong>: Permet de comparer les informations du client par rapport à l'ensemble de la base client.</p>
    
        <p><strong>Comparaison aves un groupe de clients similaires</strong>: Compare le client à un groupe de clients ayant des profils similaires pour un benchmarking ciblé.</p>
    
        <p><strong>" Descriptions des caractéristiques</strong>: Fournit des explications détaillées sur les caractéristiques utilisées dans le modèle.</p>
    

        </div>
        """

    return  explication_text