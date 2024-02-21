import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import shap
import numpy as np
import json
import seaborn as sns

from sidebar import create_sidebar

# Configuration de la page
st.set_page_config(page_title="Mon Dashboard Interactif", layout="wide")

# Créer la sidebar et récupérer les options sélectionnées
pressed, option_1, option_2, option_3, option_4, option_5, option_6, client_id_input = create_sidebar()

st.set_option('deprecation.showPyplotGlobalUse', False)

# Constante
seuil = 0.48
couleur_accepte = "#3B782F"
couleur_refuse = "#B82010"
url = f"https://ocp7-dc846df71c5b.herokuapp.com/api/predict/"
#url = f"http://127.0.0.1:5000"


        ###############################
        ###      GESTION STATE      ###
        ###############################

if 'client_id' not in st.session_state:
    st.session_state.client_id = ''

if 'message' not in st.session_state:
    st.session_state.message = ''

if not st.session_state.client_id:
    # Message d'information initial
    st.info(
        "Veuillez saisir un numéro de client dans la barre latérale à gauche et appuyer sur 'Obtenir la prédiction' pour commencer.")

def update_state(key, value):
    st.session_state[key] = value

    #######################
    ###      Jauge      ###
    #######################


def create_gauge_chart(score, decision, threshold=seuil, couleur_accepte=couleur_accepte,
                       couleur_refuse=couleur_refuse):
    # Configuration de la jauge avec les intervalles et couleurs spécifiques
    gauge = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=score,
        mode="gauge+number",
        title={'text': "Score", 'font': {'size': 24}},
        gauge={'axis': {'range': [None, threshold]},
               'bar': {'color': "grey"},
               'steps': [{'range': [0, 0.05], 'color': 'Green'},
                         {'range': [0.05, 0.098], 'color': 'LimeGreen'},
                         {'range': [0.098, 0.099], 'color': 'red'},
                         {'range': [0.1, 0.2], 'color': 'Orange'},
                         {'range': [0.2, 1], 'color': 'Crimson'}],
               'threshold': {'line': {'color': "red", 'width': 4},
                             'thickness': 1, 'value': score}
               }
    ))
    return gauge

    ##############################################
    ###         Aide Describe Client           ###
    ##############################################
def afficher_infos_client():
    html_template = """
    <div style="background-color:lightblue; padding:10px; border-radius:10px">
    <h2>Informations Client</h2>

    <p><strong>SK_ID_CURR</strong>: Identifiant unique du client. Sert de clé primaire pour identifier les enregistrements de clients de manière unique.</p>

    <p><strong>NAME_CONTRACT_TYPE</strong>: Type de contrat de crédit. Indique le type de prêt contracté par le client, par exemple, prêt à la consommation ou prêt immobilier.</p>

    <p><strong>CODE_GENDER</strong>: Genre du client. Le sexe du client, généralement indiqué par 'M' pour masculin et 'F' pour féminin.</p>

    <p><strong>CNT_CHILDREN</strong>: Nombre d'enfants. Le nombre total d'enfants que le client a.</p>

    <p><strong>AMT_INCOME_TOTAL</strong>: Revenu total annuel. Le revenu annuel total du client.</p>

    <p><strong>DAYS_BIRTH</strong>: Âge du client (en jours). L'âge du client, calculé en jours à partir de la date de naissance jusqu'à la date de la demande de prêt.</p>

    <p><strong>DAYS_EMPLOYED</strong>: Durée de l'emploi (en jours). Le nombre de jours depuis le début de l'emploi actuel du client.</p>

    </div>
    """
    st.markdown(html_template, unsafe_allow_html=True)


    ##############################################
    ###         Comparaison Groupe Client      ###
    ##############################################
def visualize_client_comparison(df, client_id, features, approval_status):

    # Transformation des données pour le graphique
    melted_data = df.melt(id_vars=['TARGET'], value_vars=features, var_name="Features", value_name="Normalized Values")

    # Création du graphique
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=melted_data, x='Features', y='Normalized Values', hue='TARGET', linewidth=1, showfliers=False,
                width=0.4, palette=['tab:green', 'tab:red'], saturation=0.5, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel("Valeurs Normalisées", fontsize=15)

    # Extraction des données du client sélectionné
    client_data = df.loc[client_id][features].to_frame().reset_index()
    client_data_renamed = client_data.rename(columns={"index": "Feature", client_id: "Value"})
    client_data_renamed['Color'] = 'green' if approval_status == 'Granted' else 'red'

    # Ajout du client au graphique
    sns.swarmplot(data=client_data_renamed, x='Feature', y='Value', linewidth=1, color=client_data_renamed['Color'].unique()[0],
                  marker='p', size=8, edgecolor='k', label='Client', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    ax.set(xlabel="", ylabel="Valeurs Normalisées")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles[:3], ["Accordé", "Refusé", "Client"])

    plt.tight_layout()
    plt.show()




def plot_feature_importances(df):
    # Sort features according to importance
    df = df.sort_values('Importance', ascending=False).reset_index(drop=True)
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['Importance'] / df['Importance'].sum()
    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(14, 10))
    ax = plt.subplot()
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:12]))),
            df['importance_normalized'].head(12),
            align='center', edgecolor='k')
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:12]))))
    ax.set_yticklabels(df['Feature'].head(12))
    # Plot labeling
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importances globales des 12 premières caractéristiques')


if pressed:
    update_state('client_id', client_id_input)
    client_id = st.session_state.client_id

    st.header('Impact Local des Caractéristiques sur la Prédiction')
    # Requête à l'API Flask
    api_url = f"{url}/api/predict/{client_id}"

    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        update_state('data', data)

        if data['classe'] == 'accepte':
            color = couleur_accepte
            message = 'accepté'
        else:
            color = couleur_refuse
            message = 'refusé'

        update_state('message', message)
        update_state('prob', data['probability'])
        update_state('color', color)

        prob = float(data['probability']) / 100
        update_state('prob', prob)

        st.write(
            f'<p style="background-color:{color}; color:#ffffff; font-size:24px; border-radius:5px; padding:10px; text-align:center;">Le prêt est {message}</p>',
            unsafe_allow_html=True)

        fig = create_gauge_chart(prob, message)

        left_col, center_col, right_col = st.columns([1, 6, 1])  # Ajustez les proportions selon besoin

        with center_col:  # Utiliser la colonne centrale pour afficher le graphique
            st.plotly_chart(fig, use_container_width=True)  # Affichez le graphique centré

        prediction_details = {
            "Probabilité de Non-Remboursement": f"{data['probability']}%",
            "Classe": data['classe']
        }

        st.markdown("---")

    # Affichage des détails de la prédiction dans des colonnes alignées sur la largeur
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("## Détail de la prédiction :")

        with col2:
            st.markdown("### Informations clés")
            for key, value in prediction_details.items():
                st.markdown(f"**{key}:** {value}")

    elif response.status_code == 404:
        st.warning("ID inconnu.")
    else:
        st.warning("Erreur lors de la requête à l'API.")

if option_1:
    if 'client_id' in st.session_state:
        st.header('Informations Client')
        client_id = st.session_state.client_id

        # Requête à l'API Flask
        api_url = f"{url}/api/data/{client_id}"
        # st.info(f"L'URL de l'API est : {api_url}")

        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()

            # Transformer les données en DataFrame
            df = pd.DataFrame(data)

            # Vous pourriez vouloir sélectionner quelques colonnes clés à afficher
            # Pour cet exemple, je vais choisir quelques colonnes arbitrairement
            columns_to_display = [
                "SK_ID_CURR", "NAME_CONTRACT_TYPE", "CODE_GENDER",
                "CNT_CHILDREN", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED"
            ]
            # Afficher le DataFrame dans Streamlit
            st.dataframe(df[columns_to_display])

            afficher_infos_client()
        else:
            st.warning("Veuillez d'abord obtenir une prédiction en validant un ID client.")

    elif response.status_code == 404:
        st.write("ID inconnu.")
    else:
        st.write("Erreur lors de la requête à l'API.")

def display_shap_plot(shap_values, expected_value, feature_names, title, nb_features):
    expl = shap.Explanation(values=np.array(shap_values).reshape(1, -1),
                            base_values=expected_value,
                            feature_names=feature_names)
    shap.plots.waterfall(expl[0], max_display=nb_features, show=False)
    plt.title(title)
    st.pyplot(bbox_inches='tight', clear_figure=True)


if option_2:
    if 'client_id' in st.session_state:
        st.header('Impact Local des Caractéristiques sur la Prédiction')
        client_id = st.session_state.client_id

        # Requête à l'API Flask
        api_url = f"{url}/api/shap/{client_id}"
        #st.info(f"L'URL de l'API est : {api_url}")

        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()

            nb_features = st.slider('Nombre de variables à visualiser', 0, 20, 10)

            st.subheader("Classe 0: Remboursement")
            display_shap_plot(data["shap_values_class_0"], data["expected_value_class_0"],
            data["feature_names"], "SHAP - Classe 0 (Remboursement)", nb_features)

            # Affichage du graphique pour la classe 1 (Non-Remboursement)
            st.subheader("Classe 1: Non-Remboursement")
            display_shap_plot(data["shap_values_class_1"], data["expected_value_class_1"],
            data["feature_names"], "SHAP - Classe 1 (Non-Remboursement)", nb_features)

            st.markdown("""
                    <style>
                    .custom-info {
                        font-size: 18px;  
                        color: #000000; 
                        background-color: #ADD8E6; 
                        padding: 10px; 
                        border-radius: 5px; 
                    }
                    </style>
                    <div class="custom-info">
                    <strong>Explication du graphique en cascade SHAP</strong><br>
                    Le graphique en cascade ci-dessus débute par la valeur attendue de la sortie du modèle, c'est-à-dire la moyenne des prédictions pour l'ensemble des observations. Cette valeur de base est le point de départ pour comprendre l'influence de chaque caractéristique sur la prédiction finale. Chaque barre du graphique, qu'elle soit orientée vers le haut (en rouge) pour une influence positive, ou vers le bas (en bleu) pour une influence négative, représente la contribution spécifique de chaque caractéristique à l'écart entre la valeur attendue et la prédiction finale pour le dossier spécifique du client. Cette visualisation permet de décomposer de manière intuitive comment chaque facteur contribue à la décision finale du modèle, en fournissant une compréhension approfondie de la prédiction à un niveau individuel.
                    </div>
                """, unsafe_allow_html=True)


        # shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_values})
        # st.write(shap_df)
        else:
            st.warning("Veuillez d'abord obtenir une prédiction en validant un ID client.")

    elif response.status_code == 404:
        st.write("ID inconnu.")
    else:
        st.write("Erreur lors de la requête à l'API.")

if option_3:
    if 'client_id' in st.session_state:
        st.header('Impact Global des Caractéristiques sur la Prédiction')
        client_id = st.session_state.client_id

        # Requête à l'API Flask
        api_url = f"{url}/api/global"
        #st.info(f"L'URL de l'API est : {api_url}")

        response = requests.get(api_url)

        if response.status_code == 200:


            data = response.json()

            features = pd.Series(data["feat_imp_global"]["Feature"]).values
            importances = pd.Series(data["feat_imp_global"]["Importance"]).values

            # Création d'un DataFrame pour les valeurs SHAP (simu
            df_shap_values = pd.DataFrame(importances.reshape(1, -1), columns=features)

            # Création d'un DataFrame pour les valeurs X
            X_val_scaled = pd.DataFrame(0, index=[0], columns=features)

            shap_values = df_shap_values.values

            left_col, center_col, right_col = st.columns([1, 6, 1])

            with center_col:
                plt.title("Importance des caractéristiques", fontsize=14)
                shap.summary_plot(shap_values, X_val_scaled, plot_type='bar', plot_size=(15, 6))
                st.pyplot(plt)


            # Afficher le graphique dans Streamlit

            st.markdown("""
                    <style>
                    .custom-info-global {
                        font-size: 18px;  
                        color: #000000; 
                        background-color: #ADD8E6; 
                        padding: 10px; 
                        border-radius: 5px; 
                    }
                    </style>
                    <div class="custom-info-global">
                    <strong>Compréhension de l'importance globale des caractéristiques</strong><br>
                    Ce graphique illustre l'importance globale des différentes caractéristiques prises en compte par le modèle pour effectuer ses prédictions. Chaque barre représente une caractéristique différente et son poids relatif dans la décision du modèle, permettant d'identifier les facteurs ayant le plus fort impact sur les prédictions, indépendamment des cas individuels. Cette vue d'ensemble aide à comprendre quelles caractéristiques le modèle considère comme les plus déterminantes pour évaluer le risque, offrant ainsi une vision claire de la logique suivie par le modèle dans son ensemble.
                    </div>
                """, unsafe_allow_html=True)

        else:
            st.warning("Veuillez d'abord obtenir une prédiction en validant un ID client.")

    elif response.status_code == 404:
        st.write("ID inconnu.")
    else:
        st.write("Erreur lors de la requête à l'API.")

if option_4:
    if 'client_id' in st.session_state:
        st.header('Comparaison avec les groupes clients')
        st.write(f"Client ID avec option_4 cochée : {st.session_state.client_id} : {st.session_state.message} : {st.session_state.prob}")
        client_id = st.session_state.client_id
        message = st.session_state.message
        prob = st.session_state.prob
        decision = 0 if message == 'accepté' else 1

        # Requête à l'API Flask
        api_url = f"{url}/api/data/knn//{client_id}"
        api_url = f"{url}/api/data/all"
        st.info(f"L'URL de l'API est : {api_url}")

        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            data_train_json = data['data_train']
            data_test_json  = data['data_test']

            df_train = pd.DataFrame.from_dict(data_train_json)
            df_train = df_train.rename(columns={'Unnamed: 0': 'SK_ID_CURR'})
            df_train.set_index('SK_ID_CURR', inplace=True)

            client_id = 47531
            seuil = 0.48

            if 47531 in df_train.index:
                st.write("La valeur 447596 est présente dans l'index de votre DataFrame.")
            else:
                st.write("La valeur 447596 n'est pas présente dans l'index de votre DataFrame.")



            # Convertir les valeurs décimales en 0 ou 1 en fonction du seuil
            df_train['TARGET'] = df_train['TARGET'].apply(lambda x: 1 if x > seuil else 0)

            if client_id in df_train.index:
                st.write(f"Client ID {client_id} exists in the DataFrame.")
            else:
                st.write(f"Client ID {client_id} does not exist in the DataFrame.")


            df_test = pd.DataFrame.from_dict(data_test_json)

            st.dataframe(df_train.sample(5))
            st.dataframe(df_test.sample(5))

            if 'TARGET' in df_train.columns:
                st.write("Colonne 'target':")
                st.write(df_train['TARGET'])
            else:
                st.error("La colonne 'target' n'est pas présente dans le DataFrame.")



            features_select = [
                "CODE_GENDER",
                "FLAG_OWN_CAR",
                "FLAG_OWN_REALTY",
                "CNT_CHILDREN",
                "AMT_INCOME_TOTAL",
                "AMT_CREDIT"
            ]

            features_num = df_train.select_dtypes(include=np.number).columns.tolist()
            features_existantes = [col for col in features_select if col in features_num]



            #features_num.remove('class')
            # features_num.remove('score')

            st.info("Comparaison avec un groupe des clients")
            st.write("Aperçu des données sur un groupe représentatif de 5 clients")

            selected_features = st.multiselect("Sélectionner les caractéristiques numériques à visualiser des clients:",
                                               options=sorted(features_existantes),
                                               default=features_existantes)

            ## Création du graphique
            fig = visualize_client_comparison(df_train, client_id, selected_features, message)
            st.pyplot(fig)

            if st.checkbox('Analyse bivariée des clients'):
                bi_select = features_select

                bi_select.insert(0, '<Select>')
                list_x = bi_select
                list_y = bi_select
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    a = st.selectbox("Sélectionner une caractéristique X ", list_x)
                with c2:
                    b = st.selectbox("Sélectionner une caractéristique Y", list_y)
                if (a != '<Select>') & (b != '<Select>'):
                    fig = px.scatter(df_train, x=a, y=b, color= 'TARGET', opacity=0.5, width=1000, height=600,
                                     color_discrete_sequence=["red", "green"],
                                     title="Analyse bivariée des caractéristiques sélectionnées pour un groupe des clients")
                    df_cust = df_train.iloc[df_train.index == str(client_id)]
                    fig.add_trace(go.Scatter(x=df_cust[a], y=df_cust[b], mode='markers',
                                             marker_symbol='star', marker_size=30, marker_color="black",
                                             name="Client"))  # showlegend=False))
                    fig.update_layout(
                        font_family="Arial",
                        font_size=15,
                        title_font_color="blue")

                    st.plotly_chart(fig, use_container_width=False)

        else:
            st.warning("Veuillez d'abord obtenir une prédiction en validant un ID client.")

    elif response.status_code == 404:
        st.write("ID inconnu.")
    else:
        st.write("Erreur lors de la requête à l'API.")

if option_6:

    api_url = f"{url}/api/desc/all"
    st.info(f"L'URL de l'API est : {api_url}")

    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()['desc']
        description = data['Description']
        row = data['Row']

        options = [(row[str(i)], description[str(i)]) for i in range(len(description))]

        # Utilisez la liste de tuples pour alimenter le selectbox
        selected_option = st.selectbox("Choisir une caractéristique :", options, format_func=lambda x: x[0])

        # Affichez la description pour l'option sélectionnée
        if selected_option:
            st.write(f"**Description:** {selected_option[1]}")
        else:
            st.error("Impossible de charger les descriptions depuis l'API.")

        show_all = st.checkbox("Tout voir")

        if show_all:
            st.markdown("""
                    <style>
                    .pair {
                        background-color: #f0f2f6; /* Couleur de fond pour les lignes paires */
                        padding: 10px;
                        border-radius: 5px;
                        margin-bottom: 5px; /* Espace entre les entrées */
                    }
                    
                    .impair {
                        background-color: #e6e6e6; /* Couleur de fond pour les lignes impaires */
                        padding: 10px;
                        border-radius: 5px;
                        margin-bottom: 5px; /* Espace entre les entrées */
                    }
                    </style>
                    """, unsafe_allow_html=True)

            # Affichage des descriptions avec style différent pour lignes paires et impaires
            for i in range(len(description)):
                div_class = "pair" if i % 2 == 0 else "impair"
                st.markdown(f"<div class='description-style'><b>{row[str(i)]}:</b> {description[str(i)]}</div>",
                            unsafe_allow_html=True)


# Histogramme
def plot_feature_distribution(df, feature):
    fig = px.histogram(df, x=feature, title=f'Distribution de {feature}')
    st.plotly_chart(fig)


# Analyse Bivariée
def plot_bivariate_analysis(df, feature1, feature2):
    fig = px.scatter(df, x=feature1, y=feature2, title=f'Analyse bi-variée entre {feature1} et {feature2}')
    st.plotly_chart(fig)


def display_prediction_graph(probability, seuil_pourc=48):
    """
    Affiche un graphique à barres indiquant la probabilité de non-remboursement.

    :param probability: La probabilité de non-remboursement exprimée en pourcentage.
    :param seuil_pourc: Le seuil de décision exprimé en pourcentage.
    """
    categories = ['Accepté', 'Refusé']
    values = [100 - probability, probability]  # Probabilités calculées
    colors = ['green', 'red']  # Couleurs pour chaque catégorie
    seuil_pourc = seuil_pourc  # Convertir le seuil en pourcentage pour correspondre à l'échelle du graphique

    # Création du graphique à barres
    fig, ax = plt.subplots()
    bars = ax.bar(categories, values, color=colors)

    # Ajout de la ligne de seuil
    ax.axhline(y=seuil_pourc, color='blue', linestyle='--', label=f'Seuil: {seuil_pourc}%')

    # Ajout de titres et étiquettes
    ax.set_ylabel('Pourcentage')
    ax.set_title('Répartition par classe')
    ax.set_ylim(0, 100)
    ax.legend()

    # Afficher les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Décalage vertical pour le label
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Retourner le graphique pour affichage dans Streamlit
    return fig



import streamlit as st



