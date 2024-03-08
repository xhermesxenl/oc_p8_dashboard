import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import shap
import numpy as np
import seaborn as sns

from aide_texte import aide_comparaison_impact, aide_comparaison_bi, afficher_infos_client, afficher_infos_local, afficher_infos_global, afficher_menu_client
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
url = f"https://ocp8api-c762537c6d53.herokuapp.com"


###############################
###      GESTION STATE      ###
###############################

if 'client_id' not in st.session_state:
    st.session_state.client_id = ''
    explication_text = afficher_menu_client()
    st.markdown(explication_text, unsafe_allow_html=True)

if 'message' not in st.session_state:
    st.session_state.message = ''



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
    ###         Feature Global                 ###
    ##############################################
def display_shap_plot(shap_values, expected_value, feature_names, title, nb_features):
        expl = shap.Explanation(values=np.array(shap_values).reshape(1, -1),
                                base_values=expected_value,
                                feature_names=feature_names)
        shap.plots.waterfall(expl[0], max_display=nb_features, show=False)
        plt.title(title)
        st.pyplot(bbox_inches='tight', clear_figure=True)




    ##############################################
    ###         Comparaison Groupe Client      ###
    ##############################################
def visualize_client_comparison(df, client_id, features, etat_approbation):

    # Préparation des données pour la visualisation
    donnees_fusionnees = df.melt(id_vars='TARGET', value_vars=features,
                                 var_name='Caractéristique', value_name='Valeur')

    # Création de la figure et de l'axe
    plt.figure(figsize=(15, 10))
    ax = sns.boxplot(data=donnees_fusionnees, x='Caractéristique', y='Valeur', hue='TARGET',
                     palette='coolwarm', fliersize=0, width=0.5)

    # Mise en forme des axes et titres
    ax.set_title('Comparaison des Caractéristiques par Statut de Prêt', fontsize=18, fontweight='bold')
    ax.set_xlabel('Caractéristique', fontsize=14)
    ax.set_ylabel('Valeur Normalisée', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    sns.despine(trim=True)

    # Extraction et mise en évidence des données du client
    valeurs_client = df.loc[[client_id], features].T.reset_index()
    valeurs_client.columns = ['Caractéristique', 'Valeur']
    valeurs_client['Couleur'] = 'approved' if etat_approbation == 'Accordé' else 'denied'
    couleur = 'green' if etat_approbation == 'Accordé' else 'red'

    sns.scatterplot(data=valeurs_client, x='Caractéristique', y='Valeur', s=200, color='black', ax=ax, zorder=5, label='Client')

    # Amélioration de la légende
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:2] + [handles[-1]], labels=['Refusé', 'Accordé', 'Client'], title='Statut', frameon=False)

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

def modifier_infos_client(client_id, data_json):
    api_url = f"{url}/api/predictpost"

    response = requests.post(api_url, json=data_json)

    if response.status_code == 200:
        data = response.json()
    else :
        data = {"status": "Ko", "code": response.status_code}

    return data




if pressed:
    update_state('client_id', client_id_input)
    client_id = st.session_state.client_id

    st.header('Prédiction')
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

        st.info("Visualisez les informations détaillées des clients ci-dessous. Vous avez également la possibilité de mettre à jour les valeurs spécifiques à chaque client en activant l'option de modification.")


        # Requête à l'API Flask
        api_url = f"{url}/api/data/{client_id}"

        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()

            # Transformer les données en DataFrame
            df = pd.DataFrame(data)

            

            # Colonnes à afficher avec les noms conviviaux
            columns_mapping = {
                "SK_ID_CURR": "ID Client",
                "NAME_CONTRACT_TYPE": "Type de Contrat",
                "CODE_GENDER": "Genre",
                "CNT_CHILDREN": "Nombre d'Enfants",
                "AMT_INCOME_TOTAL": "Revenu Total",
                "DAYS_BIRTH": "Âge (Année)",
                "DAYS_EMPLOYED": "Emploi (jours)",
                "EXT_SOURCE_3": "Source Externe 3",
                "EXT_SOURCE_2": "Source Externe 2",
                "AMT_CREDIT": "Montant du Crédit"
            }

            columns_to_display = [
                "ID Client", "Type de Contrat", "Genre",
                "Nombre d'Enfants", "Revenu Total", "Âge (Année)", "Emploi (jours)"
            ]

            # Mise à jour des noms des colonnes dans le DataFrame pour l'affichage
            df_for_display = df.rename(columns=columns_mapping)
            st.dataframe(df_for_display[columns_to_display])


            columns_to_update = [
                "EXT_SOURCE_3", "EXT_SOURCE_2", "AMT_CREDIT", "AMT_INCOME_TOTAL"
            ]

            columns_to_modify = [col for col in columns_to_update if col != "SK_ID_CURR"]

            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)

            modifications = {}

            st.info("Activer la modification des informations client")
            if st.checkbox(''):
                for col in columns_to_modify:
                    # Afficher un champ de saisie pour chaque colonne, avec la valeur actuelle comme placeholder
                    current_value = df[col].iloc[0]
                    if isinstance(current_value, float):
                        if current_value.is_integer():
                            placeholder_value = str(int(current_value))
                        else:
                            placeholder_value = f"{current_value:.2f}"
                    else:
                        placeholder_value = str(current_value)

                    new_value_str = st.text_input(f"{col}", placeholder_value, key=f"new_{col}")




                    try:
                        new_value = float(new_value_str)

                        if new_value.is_integer():
                            # Si la nouvelle valeur est un entier, enlever les décimales
                            new_value_formatted = str(int(new_value))
                        else:
                            # Si c'est un nombre décimal, garder deux décimales
                                            new_value_formatted = f"{new_value:.2f}"
                    except ValueError:
                        new_value = new_value_str

                    if new_value_str != placeholder_value:
                        # df.at[0, col] = new_value  # Mettre à jour la valeur dans le DataFrame
                        modifications[col] = new_value_str




                if st.button("Valider les modifications"):

                    if modifications:
                             data_to_send = {
                            "id_client": client_id,
                            "data": modifications
                        }

                    data = modifier_infos_client(client_id, data_to_send)

                    prediction_details_new = {
                        "Probabilité de Non-Remboursement": f"{data['probability']}%",
                        "Classe": data['classe']
                    }

                    ## Prediction precedente :
                    message = st.session_state.message
                    prob = st.session_state.prob
                    prob = round(float(prob) * 100)
                    prediction_details_old = {
                        "Probabilité de Non-Remboursement": f"{prob}%",
                        "Classe": message
                    }

                    # Affichage des détails de la prédiction dans des colonnes alignées sur la largeur
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("## Détail de la nouvelle prédiction :")
                    with col2:
                        st.markdown("### Informations clés")
                        for key, value in prediction_details_new.items():
                            st.markdown(f"**{key}:** {value}")

                    st.success("Modifications enregistrées et prédiction mise à jour.")

                    with col1:
                        st.markdown("## Détail de l'ancienne prédiction :")
                    with col2:
                        st.markdown("### Informations clés")
                        for key, value in prediction_details_old.items():
                            st.markdown(f"**{key}:** {value}")

                st.markdown('<br>', unsafe_allow_html=True)
                st.markdown('<br>', unsafe_allow_html=True)
                explication_text = afficher_infos_client()
                st.markdown(explication_text, unsafe_allow_html=True)

        else:
            st.warning("Veuillez d'abord obtenir une prédiction en validant un ID client.")

    elif response.status_code == 404:
        st.write("ID inconnu.")
    else:
        st.write("Erreur lors de la requête à l'API.")





if option_2:
    if 'client_id' in st.session_state:
        st.header('Impact Local des Caractéristiques sur la Prédiction')
        client_id = st.session_state.client_id

        # Requête à l'API Flask
        api_url = f"{url}/api/shap/{client_id}"
        #st.info(f"L'URL de l'API est : {api_url}")

        response = requests.get(api_url)

        st.info("Personnalisez votre exploration : Utilisez le slider ci-dessous pour sélectionner le nombre de variables que vous souhaitez visualiser. Cet outil interactif vous permet d'ajuster la profondeur de l'analyse à votre convenance, mettant en lumière les facteurs les plus influents sur la prédiction de risque selon votre sélection.")
        if response.status_code == 200:
            data = response.json()

            st.markdown("""
            <style>
            .big-font {
                font-size:20px !important;
            }
            </style>
            <div class='big-font'>Nombre de variables à visualiser</div>
            """, unsafe_allow_html=True)

            # Création du slider
            nb_features = st.slider('', 0, 20, 10)



            st.subheader("Classe 0: Remboursement")
            display_shap_plot(data["shap_values_class_0"], data["expected_value_class_0"],
                              data["feature_names"], "SHAP - Classe 0 (Remboursement)", nb_features)

            # Affichage du graphique pour la classe 1 (Non-Remboursement)
            st.subheader("Classe 1: Non-Remboursement")
            display_shap_plot(data["shap_values_class_1"], data["expected_value_class_1"],
                              data["feature_names"], "SHAP - Classe 1 (Non-Remboursement)", nb_features)


            # Afficher l'aide
            explication_text =  afficher_infos_local()
            st.markdown(explication_text, unsafe_allow_html=True)

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


            # Afficher l'aide
            explication_text =  afficher_infos_global()
            st.markdown(explication_text, unsafe_allow_html=True)

        else:
            st.warning("Veuillez d'abord obtenir une prédiction en validant un ID client.")

    elif response.status_code == 404:
        st.write("ID inconnu.")
    else:
        st.write("Erreur lors de la requête à l'API.")


if option_4:
    if 'client_id' in st.session_state:
        st.header('Comparaison avec les groupes clients')
        # st.write(f"Client ID avec option_4 cochée : {st.session_state.client_id} : {st.session_state.message} : {st.session_state.prob}")
        client_id = st.session_state.client_id
        client_id = int(client_id)
        message = st.session_state.message
        prob = st.session_state.prob

        decision = 0 if message == 'accepté' else 1

        # Requête à l'API Flask
        api_url = f"{url}/api/data/all"

        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            data_train_json = data['data']
            data_info_json  = data['data_info']

            df_train = pd.DataFrame.from_dict(data_train_json)
            df_info = pd.DataFrame.from_dict(data_info_json)
            df_train.set_index('SK_ID_CURR', inplace=True)

            seuil = 0.48

            # Convertir les valeurs décimales en 0 ou 1 en fonction du seuil
            df_train['TARGET'] = df_train['TARGET'].apply(lambda x: 1 if x > seuil else 0)

            st.info("Aperçu des données sur un groupe représentatif de 5 clients")
            st.dataframe(df_info.sample(5))

            features_select = [
                "SK_ID_CURR", "NAME_CONTRACT_TYPE", "CODE_GENDER",
                "CNT_CHILDREN", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED"
            ]

            features_num = df_train.select_dtypes(include=np.number).columns.tolist()
            features_existantes = [col for col in features_select if col in features_num]


            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)
            st.info("Sélectionner les caractéristiques numériques à visualiser des clients:")

            # Création du multiselect
            selected_features = st.multiselect("",
                                                   options=sorted(features_existantes),
                                                   default=features_existantes)

            ## Création du graphique
            fig = visualize_client_comparison(df_train, client_id, selected_features, message)
            st.pyplot(fig)

            ## Aide
            explication_text = aide_comparaison_impact()
            st.markdown(explication_text, unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)

            st.info("Analyse bivariée des clients : Cochez la case ci-dessous pour une analyse bivariée approfondie, révélant les interactions clés entre les caractéristiques des clients qui influencent le risque de crédit.")

            if st.checkbox(''):
                bi_select = features_select

                options_selection = ['<Choisir>'] + bi_select

                # Définition des variables pour les sélections X et Y
                selection_X = options_selection
                selection_Y = options_selection

                # Création de colonnes pour les sélections
                col1, col2, col3, col4 = st.columns(4)

                # Sélection de la caractéristique X
                with col2:
                    choix_X = st.selectbox("Choix caractéristique X", selection_X)

                # Sélection de la caractéristique Y
                with col3:
                    choix_Y = st.selectbox("Choix caractéristique Y", selection_Y)

                # Vérification que les choix ne sont pas l'option par défaut
                if (choix_X != '<Choisir>') and (choix_Y != '<Choisir>'):
                    # Création du graphique
                    graphique = px.scatter(df_train, x=choix_X, y=choix_Y, color='TARGET', opacity=0.6, width=1000, height=600,
                                           color_discrete_map={"1": "red", "0": "green"},
                                           title="Analyse des variables choisies pour un ensemble de clients")

                    # Ajout d'un marqueur pour le client spécifique
                    client_specifique = df_train.loc[df_train.index == str(client_id)]
                    graphique.add_trace(go.Scatter(x=client_specifique[choix_X], y=client_specifique[choix_Y], mode='markers',
                                                   marker=dict(symbol='star', size=20, color="black"),
                                                   name="Client sélectionné"))

                    # Mise à jour de la configuration du graphique
                    graphique.update_layout(
                        font=dict(family="Courier New", size=14, color="navy"),
                        title_font=dict(size=16, color="darkred"))

                    # Affichage du graphique
                    st.plotly_chart(graphique, use_container_width=True)

                    # Aide
                    explication_text = aide_comparaison_bi(choix_X,choix_Y)
                    st.markdown(explication_text, unsafe_allow_html=True)

        else:
            st.warning("Veuillez d'abord obtenir une prédiction en validant un ID client.")

    elif response.status_code == 404:
        st.write("ID inconnu.")
    else:
        st.write("Erreur lors de la requête à l'API.")



if option_5:
    if 'client_id' in st.session_state:
        st.header('Comparaison avec un groupes clients similaires')
        client_id = st.session_state.client_id
        client_id = int(client_id)
        message = st.session_state.message
        prob = st.session_state.prob

        decision = 0 if message == 'accepté' else 1

        # Requête à l'API Flask
        api_url = f"{url}/api/data/knn/{client_id}"

        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            data_train_json = data['df_sim_client_norm']
            data_info_json  = data['df_sim_client']

            df_train = pd.DataFrame.from_dict(data_train_json)
            df_info = pd.DataFrame.from_dict(data_info_json)
            df_train.set_index('SK_ID_CURR', inplace=True)

            seuil = 0.48


            # Convertir les valeurs décimales en 0 ou 1 en fonction du seuil
            df_train['TARGET'] = df_train['TARGET'].apply(lambda x: 1 if x > seuil else 0)

            st.info("Aperçu des données sur un groupe de 5 clients similaires")
            st.dataframe(df_info.sample(5))

            features_select = [
                "SK_ID_CURR", "NAME_CONTRACT_TYPE", "CODE_GENDER",
                "CNT_CHILDREN", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED"
            ]

            features_num = df_train.select_dtypes(include=np.number).columns.tolist()
            features_existantes = [col for col in features_select if col in features_num]

            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)
            st.info("Sélectionner les caractéristiques numériques à visualiser des clients:")

            # Création du multiselect
            selected_features = st.multiselect("",
                                               options=sorted(features_existantes),
                                               default=features_existantes)

            ## Création du graphique
            fig = visualize_client_comparison(df_train, client_id, selected_features, message)
            st.pyplot(fig)

            ## Aide
            explication_text = aide_comparaison_impact()
            st.markdown(explication_text, unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)

            st.info("Analyse bivariée des clients : Cochez la case ci-dessous pour une analyse bivariée approfondie, révélant les interactions clés entre les caractéristiques des clients qui influencent le risque de crédit.")

            if st.checkbox(''):

                bi_select = features_select

                options_selection = ['<Choisir>'] + bi_select

                # Définition des variables pour les sélections X et Y
                selection_X = options_selection
                selection_Y = options_selection

                # Création de colonnes pour les sélections
                col1, col2, col3, col4 = st.columns(4)

                # Sélection de la caractéristique X
                with col2:
                    choix_X = st.selectbox("Choix caractéristique X", selection_X)

                # Sélection de la caractéristique Y
                with col3:
                    choix_Y = st.selectbox("Choix caractéristique Y", selection_Y)

                # Vérification que les choix ne sont pas l'option par défaut
                if (choix_X != '<Choisir>') and (choix_Y != '<Choisir>'):
                    # Création du graphique
                    graphique = px.scatter(df_train, x=choix_X, y=choix_Y, color='TARGET', opacity=0.6, width=1000, height=600,
                                           color_discrete_map={"1": "red", "0": "green"},
                                           title="Analyse des variables choisies pour un ensemble de clients")

                    # Ajout d'un marqueur pour le client spécifique
                    client_specifique = df_train.loc[df_train.index == str(client_id)]
                    graphique.add_trace(go.Scatter(x=client_specifique[choix_X], y=client_specifique[choix_Y], mode='markers',
                                                   marker=dict(symbol='star', size=20, color="black"),
                                                   name="Client sélectionné"))

                    # Mise à jour de la configuration du graphique
                    graphique.update_layout(
                        font=dict(family="Courier New", size=14, color="navy"),
                        title_font=dict(size=16, color="darkred"))

                    # Affichage du graphique
                    st.plotly_chart(graphique, use_container_width=True)

                    # Aide
                    explication_text = aide_comparaison_bi(choix_X,choix_Y)
                    st.markdown(explication_text, unsafe_allow_html=True)
        else:
            st.warning("Veuillez d'abord obtenir une prédiction en validant un ID client.")

    elif response.status_code == 404:
        st.write("ID inconnu.")
    else:
        st.write("Erreur lors de la requête à l'API.")

if option_6:

    api_url = f"{url}/api/desc/all"
    #st.info(f"L'URL de l'API est : {api_url}")

    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()['desc']
        description = data['Description']
        row = data['Row']

        options = [(row[str(i)], description[str(i)]) for i in range(len(description))]

        # Utilisez la liste de tuples pour alimenter le selectbox
        st.info(f"Choisir une caractéristique afin de voir sa description:")

        selected_option = st.selectbox("", options, format_func=lambda x: x[0])

        # Affichez la description pour l'option sélectionnée
        if selected_option:
            st.warning(f"**Description:** {selected_option[1]}")
        else:
            st.error("Impossible de charger les descriptions depuis l'API.")

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)


        st.info("Afficher toutes les descriptions")

        if st.checkbox(""):
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



