import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import numpy as np
from streamlit import columns, subheader
#from streamlit import meu_label
from adjustText import adjust_text
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import LabelEncoder
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessário para gráficos 3D
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

#Anova
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import os

st.write("Python em uso:", sys.executable)

st.title( 'Análise das componentes principais (PCA)')
st.write('Email: jose.g.oliveira@ufv.br')

tab1, tab2, tab3, tab4, tab5 = st.tabs( ['PCA', 'Análise de Clustering ', 'Análise hierárquica clustering', 'Dendograma hierárquico','Anova'])
with tab1:
    label1 = 'Envie o seu arquivo CSV'
    arquivo = st.file_uploader(label1, type="csv")

    if arquivo is  None:
        st.warning('Aguardando a escolha dos dados ')

    else:
        st.success(f"O arquivo selecionado foi: {arquivo.name}")
        data = pd.read_csv(arquivo, encoding='UTF-8', sep=';')

        escolha1 = st.radio("Você deseja ver seus dados ?", ["Sim", "Não"])
        if escolha1=='Sim':
            st.dataframe(data)
        data1 = data.to_dict()
        chaves = data1.keys()
        chaves1 = list(chaves)

        #1.0 Selecionar as variáveis categóricas de interesse
        options = st.multiselect('Selecione todas as  variáveis categóricas ',['Selecione'] + chaves1, key = 'N_1')
        #1.1.Identificar e contabilizar NA
        if options:
            var1 = options
            # transformando esses dados em array numpy
            # print(escolha_categoria)
            #data1 = data.drop(columns = options) # estou conseguindo contar as NA's e retirar as Na's sem tirar a variável categórica
            data1 = data
            data2 = data1.isna().sum()
            data3 = data1


            # fazer uma função aqui! def retirarana(data na , data) e chamo novamente em outro lugar
            if data2.sum() == 0:
                st.write(f'Você Não tem **NA** nas  variáveis de seus dados  ')
                st.dataframe(data2)
                n_linhas = data3.shape[0]

                n_colunas = data3.shape[1]
                categorica = len(options)  # Número de variáveis categóricas
                # categoricas = data.shape[1]
                continua = n_colunas - categorica
                st.success(f'Número de observações: {n_linhas}')
                st.success(f'Número de variáveis: {n_colunas}')
                st.success(f'Número de variáveis categóricas:{categorica}')
                st.success(f'Número de variáveis contínuas:{continua}')

            else:
                st.write(f'Você  tem **NA** nas  variáveis de seus dados  ')
                st.dataframe(data2)
                st.write('Você gostaria de retira  as **NAs** ou substituir por valores médios?')
                escolha_2 = st.radio("Você deseja ?", ["Substituir por Valores médios", "Retirar Na"], horizontal = True )
                if escolha_2 == "Substituir por Valores médios":
                    data3 = data3.fillna(data3.median(numeric_only=True))
                    st.dataframe(data3)
                    n_linhas = data3.shape[0]

                    n_colunas = data3.shape[1]
                    categorica = len(options)  # Número de variáveis categóricas
                    # categoricas = data.shape[1]
                    continua = n_colunas - categorica
                    st.success(f'Número de observações: {n_linhas}')
                    st.success(f'Número de variáveis: {n_colunas}')
                    st.success(f'Número de variáveis categóricas:{categorica}')
                    st.success(f'Número de variáveis contínuas:{continua}')
                else:
                    st.write('Retirados NAs')
                    data3 = data3.dropna(axis=1)
                    st.dataframe(data3)
                    n_linhas = data3.shape[0]
                    n_colunas = data3.shape[1]
                    categorica = len(options)  # Número de variáveis categóricas
                    # categoricas = data.shape[1]
                    continua = n_colunas - categorica  # manter o mes
                    st.success(f'Número de observações: {n_linhas}')
                    st.success(f'Número de variáveis: {n_colunas}')
                    st.success(f'Número de variáveis categóricas:{categorica}')
                    st.success(f'Número de variáveis contínuas:{continua}')






            #2.0.Normalização dos dados (variáveis contínuas)
            st.subheader('1° Etapa da PCA (Normalização')
            st.write('Objetivo é o de padronizar todas as variáveis com a média igual a zero o desvi padrão igual a 1 ')
            st.write('Dessa forma não haverá valores discrepantes')

            st.write( 'Fórmula = (X -Xmean)/std')
            data4 = data3.drop(columns = options)# Para calcular pela biblioteca numpy
            mean = np.mean(data4, axis = 0)
            std = np.std(data4, axis = 0)
            x = ((data4 - mean)/std)
            st.dataframe(x)
            st.subheader("Média = 0 e std = 1 ")

            colunas = x.columns# pegar o nome de las colunas


            data_grouped = x[colunas].describe().T# consegui transpor a matriz, para que as variáveis virem colunas
            st.dataframe(data_grouped)
            # construindo a PCA
            #1 qual é o melhor número de componentes?
            n_linhas = data3.shape[0]
            n_colunas = data3.shape[1]
            categorica
            if n_linhas > n_colunas:
                n_linhas = n_colunas-categorica


            #t.write(n_linhas)#499 variáveis

            #3.0 processo de identificação de número de PCA's, pelo elbow methd
            SSD = []
            st.write(n_linhas)
            for k in range(n_linhas):# o número de PCA máximo deve ser igual ao número de observações
                pca = PCA(n_components = k)
                X_pca = pca.fit_transform(x)
                explained_variance = pca.explained_variance_ratio_
                SSD_PCA = np.sum(explained_variance)
                SSD.append(SSD_PCA)
            #gráfico:
            k_range = range(n_linhas)
            fig1 , ax = plt.subplots(figsize = (10,6))
            ax.plot(k_range, SSD, marker="o", linestyle="--")
            ax.set_title("Elbow Method for Optimal Number of PCA")
            ax.set_xlabel("Number of PCA")
            ax.set_ylabel("Sum of Squared Errors (SSE)")
            st.pyplot(fig1)
            a = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.90)[0][0] + 1

            st.success(f'Número de PCA que explica pelo menos 90% da variação = {a}')
            #cálculo da PCA que realmente quer
            number_pca = st.number_input('Quantas componentes principais você quer?', min_value = 2 , placeholder = 'Escolha o número ')
            st.success(f'O número escolhido foi:  {number_pca}')
            # fazendo a pca para q uantidade escolhida
            pca = PCA(n_components= number_pca)
            X_pca = pca.fit_transform(x)
            st.dataframe(X_pca)








            #Gráfico
            if number_pca==1:
                fig2, ax = plt.subplots(figsize = (10,6))
                ponto = plt.scatter(X_pca[:, 0],X_pca[:, 1], cmap='viridis', edgecolors='k', alpha=0.7)
                ax.set_xlabel("Componente Principal 1")
                ax.set_ylabel("Componente Principal 1")
                ax.set_title("PCA ")
                st.pyplot(fig2)
            elif number_pca==2:
                fig3, ax = plt.subplots(figsize=(10, 6))
                ponto = plt.scatter(X_pca[:, 0], X_pca[:, 1],  edgecolors='k', alpha=0.7)
                ax.set_xlabel("Componente Principal 1")
                ax.set_ylabel("Componente Principal 2")
                ax.set_title("PCA ")
                st.pyplot(fig3)
                # se pca=3
            elif number_pca==3:
                fig4 = plt.figure(figsize=(10, 10))
                ax = fig4.add_subplot(111, projection='3d')

                # scatter 3D
                ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],s=80, c='blue', edgecolors='k', alpha=0.7)

                # rótulos dos eixos
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')
                ax.set_title('PCA: PC1 vs PC2 vs PC3')

                plt.tight_layout()
                st.pyplot(fig4)
            else:
                st.warning('Não é possível fazer gráfico com mais de 3 dimensões')

            # Cria os índices dinamicamente de acordo com o número de componentes
            propor_autova = pd.DataFrame(
                pca.explained_variance_ratio_ ,
                index=[f'PC{i + 1}%' for i in range(len(pca.explained_variance_ratio_))],
                columns=['Autovalores'])
            st.dataframe(propor_autova)

            propor_autova1 = pd.DataFrame(
                pca.explained_variance_ratio_ * 100,
                index=[f'PC{i + 1}%' for i in range(len(pca.explained_variance_ratio_))],
                columns=['Porcentagem Autovalores']
            )
            st.dataframe(propor_autova1)





            propor_autova3 = pd.DataFrame(
            pca.components_,
            index=[f'PC{i + 1}' for i in range(len(pca.components_))],
            columns=data4.columns)
            #st.dataframe(propor_autova3)
            number_pca
            #estudar isso aqui
            for pc_name in propor_autova3.index[:number_pca]:
                # Ordena todas as variáveis dessa PC em ordem decrescente (valor absoluto)
                loadings_sorted = propor_autova3.loc[pc_name].abs().sort_values(ascending=False)

                # Se quiser mostrar os valores originais (positivos ou negativos), pode usar:
                loadings_sorted_signed = propor_autova3.loc[pc_name].loc[loadings_sorted.index]

                st.write(f" Loadings  ordenados de forma decrescente  para {pc_name}:")
                st.dataframe(loadings_sorted_signed)


            #Gráfico para saber  a porcentagem explicada pelas variáveis:
            explained_variance = np.cumsum(pca.explained_variance_ratio_)
            fig5, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
            ax.set_title('Explained Variance by Principal Components')
            ax.set_xlabel('Number of Principal Components')
            ax.set_ylabel('Cumulative Explained Variance')  # o que esse script me diz?
            st.pyplot(fig5)
            #a = np.where(np.cumsum(explained_variance) >= 0.90)[0][0] + 1
            #b = np.where(np.cumsum(explained_variance))[0][0]
            teste = len(propor_autova1)-1 # está contando 3 com o 0
            #Porcentagem explicada
            st.success(f'As {number_pca}  PCAS juntas explicam cerca de {explained_variance[teste]*100:.2f}% da variância total ')

            #preparação para o gráfico PCA
            pca = PCA(n_components= number_pca)
            X_pca = pca.fit_transform(x)# transformação dos dados para colocar no gráfico
            #loadings = pca.components_
            loading = propor_autova3
            #loading1 = loading[(loading > 0.050) & (loading <= 0.080)]
            loading1 = loading[(loading < 1) & (loading >= -1)]
            loading2 = loading1.dropna(axis=1)
            pc1 = round(propor_autova1.iloc[0,0])
            pc2 = round(propor_autova1.iloc[1,0])
            #st.write(loading1)
            #st.write(len(loading2.T))




            
            options2 = st.selectbox('Qual variável categórica você deseja ver no gráfico? ', ['Selecione'] + options,
                                    key='m_1')

            if options2 != 'Selecione':





                if number_pca==2:
                    st.subheader('Gráficos das PCAS')
                    data99 = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])  # coloquei as PC's
                    data99[options2] = data[options2]
                    st.write(data99)

                    # selecione as variáveis que você quer ver no gráfico
                    if loading2.shape[1] >1 :  # linha1 e linha2
                        st.warning(f'Você tem mais de {loading2.shape[1]} variáveis ')
                        varia1 = st.number_input('Quantas variáveis   você quer no gráfico?', min_value=2,max_value=loading2.shape[1],
                                                 placeholder='Escolha o número ', key='n_v')

                        escala = st.slider('Escala para as setas dos loadings',
                                               min_value = 1, max_value = 100,
                                               value = 1, step = 1 )

                        # st.write(loading2.iloc[:, 0:6].T)  # Transpõe depois de selecionar colunas
                        loading2 = loading2.iloc[:, 0:varia1]  # Pega as 5 primeiras colunas
                        # st.write(f'plotar : {loading2}')

                    def plot_pca(data99, loading2, escala):
                        fig, ax = plt.subplots(figsize=(10, 8))

                        sns.scatterplot(
                            data=data99,
                            x='PC1',
                            y='PC2',
                            hue=options2,  # Se `options` tiver só uma coluna
                            palette='Set1',
                            ax=ax
                        )

                        ax.set_xlabel(f'Componte principal 1 ({pc1}%) ')
                        ax.set_ylabel(f'componente principal 2 ({pc2}%)  ')

                        # Anotações (nome das variáveis)




                        texts = []
                        for i, nome_variavel in enumerate(loading2.columns):
                            x_loading = loading2.iloc[0, i]*escala
                            y_loading = loading2.iloc[1, i]*escala

                            text = ax.annotate(
                                nome_variavel,
                                (x_loading, y_loading),
                                color="black",
                                fontsize=7,
                                va='bottom',
                                rotation=45
                            )

                            texts.append(text)

                        # Ajusta todos os textos juntos
                        adjust_text(texts)

                        # Setas
                        for i, text in enumerate(texts):
                            x_pos, y_pos = text.get_position()

                            ax.arrow(
                                0, 0,
                                x_pos, y_pos,
                                color="#873600",
                                width=0.01,
                                head_width=0.05,
                                shape='full',
                                length_includes_head=True
                            )

                        return fig


                    # Exibir no Streamlit
                    fig = plot_pca(data99, loading2, escala)
                    st.pyplot(fig)

                elif number_pca == 3:
                    st.subheader('Gráfico de 3 PCA')
                    pc1 = round(propor_autova1.iloc[0, 0])
                    pc2 = round(propor_autova1.iloc[1, 0])
                    pc3 = round(propor_autova1.iloc[2, 0])
                    data89 = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])  # coloquei as PC's
                    data89[options2] = data[options2]
                    st.write(loading2)


                    if loading2.shape[1] > 2:  # linha e coluna
                        st.warning(f'Você tem {loading2.shape[1]} variáveis ')
                        varia1 = st.number_input('Quantas variáveis   você quer no gráfico?', min_value=2,
                                                 placeholder='Escolha o número ', key='n_v')
                        escala = st.slider('Escala para as setas dos loadings',
                                           min_value=1, max_value=100,
                                           value=1, step=1)

                        # st.write(loading2.iloc[:, 0:6].T)  # Transpõe depois de selecionar colunas
                        loading2 = loading2.iloc[:, 0:varia1]  # Pega as 5 primeiras colunas
                        # st.write(f'plotar : {loading2}')
                        vetores = loading2.T.values*escala# esses são os valores
                        #st.write(vetores)
                        name = loading2.T.index.tolist() # o tolist vai quebrar o Data Frame, pois ele não mais me importa
                        #st.write(name)








                    fig77 = px.scatter_3d(
                        data89,
                        x='PC1',
                        y='PC2',
                        z='PC3',
                        color=options2,
                        opacity=0.7,
                        # symbol=categorica,
                        size_max=10,
                        title=f'PCA: PC1 vs PC2 vs PC3',)

                    fig77.update_layout(
                        scene=dict(
                            xaxis_title=f'Componente principal 1 ({pc1}%)',
                            yaxis_title=f'Componente principal 2 ({pc2}%)',
                            zaxis_title=f'Componente principal 3 ({pc3}%)',
                            bgcolor='white'
                        ),
                        legend_title=options2,
                        title_font_size=18
                    )

                    for i in range(varia1):
                        fig77.add_trace(go.Scatter3d(
                            x=[0, vetores[i, 0]],
                            y=[0, vetores[i, 1]],
                            z=[0, vetores[i, 2]],
                            mode='lines+markers+text',
                            text=[None, name[i]],
                            textposition='bottom center', # 'middle right'  # ou 'top right', 'bottom center'
                            line=dict(width=4, color='black'),
                            showlegend=False
                        ))






                    # Mostra no Streamlit
                    st.plotly_chart(fig77,
                                    use_container_width=True)  # gra´fic usa toda a largyra do app, quando True


                with tab2:
                    st.header('Análise de clustering')
                    st.dataframe(x)
                    # o número  de clustering deve ser igual ao número de observações
                    SSD = []  # soma dos quadrados dos desvios, vetor para armazená-los

                    for k in range(1,n_linhas):  # testa de 1 a 76 clusters
                        kmeans = KMeans(n_clusters=k,
                                        random_state=42)  # Número de cluster vai ser definido em cada iteração , Random_state =42
                        # mesmos valores aleatórios
                        kmeans.fit(x)  # aplicação do método para cada novo número K de cluster, para todo K ele faz as 3 etapas até que o centróide seja constante
                        SSD.append(kmeans.inertia_)  # WCSS (soma dos erros quadráticos) são armazenados no vetor

                        # print('Valor da soma dos quadrados do erros(SSD) para cada número de cluster de 1 a 10\n',SSD)
                    #gr5áfico de K-cluster
                    k_range = range(1,n_linhas)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(k_range, SSD, marker="o", linestyle="--")
                    ax.set_title("Elbow Method for Optimal Number of K")
                    ax.set_xlabel("Number of K")
                    ax.set_ylabel("Sum of Squared Errors (SSE)")
                    st.pyplot(fig)

                    clustering  = st.number_input('Escolha o número de clusters', min_value=2, max_value=n_linhas, placeholder='Digite a partir de 1', key = 'clu1')
                    st.success(f'Clustering: {clustering}')
                    data_clust = x# não quero ficar sempre chamando de x
                    kmeans = KMeans(n_clusters=clustering,random_state=42).fit(x)
                    data_clust['clustering'] = kmeans.labels_
                    #Aqui é somente para calcular os valores médios da variância em função de cada cluster
                    data_clust = data_clust.join(data[options])# Aqui já coloca todas as variáveis categóricas na planilha para mim
                    st.dataframe(data_clust)





                    # pegando as pca's calculadas anteriormente
                    #Para fazer o gráfico
                    #dataclust2 =data_clust.groupby('Cluster').mean()
                    #st.dataframe(dataclust2)
                    if number_pca == 2:
                        pca = PCA(n_components=number_pca)
                        X_PCA = pca.fit_transform(x)
                        data5 = pd.DataFrame(X_PCA,columns=['PC1', 'PC2'])# coloquei as PC's

                        data5['Cluster'] = kmeans.labels_# Coloquei os clusters
                        data5 = data5.join(data[options])# coloquei as variáveis categóricas
                        #st.dataframe(data5)# Montei a tabela

                        #variância dos clusters, o cícrculo de variância
                        data6 = data5.drop(columns = options)
                        dataN = data6# tirei a variável categórica
                        data6['variancia_tot'] = data6['PC1'] + data6['PC2']# someis as variãncias de PCs
                        #st.dataframe(data6)
                        data7 = data6.drop(columns = ['PC1','PC2'])# Tirei as PC's


                        cluster_var = data7.groupby('Cluster')['variancia_tot'].std()


                        #Média de PC1 e PC2 em função dos clusters: acha-se  as coordenadas dos centróides(os centróides)
                        dataclust2 = dataN.groupby('Cluster').mean()  # são as coordenads  de centróides

                        array_cluster = dataclust2.values
                        st.dataframe(dataclust2)








                    # consigo fazer os gráfico dos meus clusters:
                        st.header('Kmeans Clustering')
                        categorica = st.selectbox('Escolha a  variável categórica que você quer ver no gráfico ', ['Selecione'] + options, key='c41')
                        if categorica !='Selecione':

                            fig51, ax = plt.subplots(figsize=(10, 6))
                            sns.scatterplot(
                                data=data5,
                                x='PC1',
                                y='PC2',
                                hue= categorica,  # Se `options` tiver só uma coluna
                                palette='Set1',
                                ax=ax
                            )
                            ax.set_xlabel(f'principal componet 1 ({pc1}%) ')
                            ax.set_ylabel(f'principalcomponent 2 ({pc2}%)')



                            for linha in range(len(array_cluster )):
                                x_coord = array_cluster [linha][0]
                                y_coord = array_cluster [linha][1]
                                raio = cluster_var.iloc[linha]*2 # Obtém o raio do cluster

                                circle2 = plt.Circle((x_coord, y_coord), raio, color='#f5eef8', fill=True,
                                                     linestyle='solid', alpha=0.5)
                                ax.add_patch(circle2)
                                # Marcador do centróide
                                ax.plot(x_coord, y_coord, 'o', color='black', markersize=2)

                            # Anotações (nome das variáveis)
                            texts = []
                            for i, nome_variavel in enumerate(loading2.columns):
                                x_loading = loading2.iloc[0, i] * escala
                                y_loading = loading2.iloc[1, i] * escala

                                text = ax.annotate(
                                    nome_variavel,
                                    (x_loading, y_loading),
                                    color="black",
                                    fontsize=7,
                                    va='bottom',
                                    rotation=45
                                )

                                texts.append(text)

                            # Ajusta todos os textos juntos
                            adjust_text(texts)

                            # Setas
                            for i, text in enumerate(texts):
                                x_pos, y_pos = text.get_position()

                                ax.arrow(
                                    0, 0,
                                    x_pos, y_pos,
                                    color="#873600",
                                    width=0.01,
                                    head_width=0.05,
                                    shape='full',
                                    length_includes_head=True
                                )



                            st.pyplot(fig51)
                            st.dataframe(data5[categorica].value_counts())
                            st.header('Métricas para os resultados de clustering ')
                            st.subheader('Silhouette score')

                            st.write('Os resultado de **silhouette score** variam de 1 a -1 ')
                            st.write('Se próximos de 1, então os clusters estão bem **separados** e **compactados** ')
                            st.write('Se próximo de zero, então os clusters podem estar **sobrepostos** ')
                            st.write('Se negativos, então os pontos estão no **local errado** ')
                            pca = PCA(n_components=number_pca)
                            X2_PCA = pca.fit_transform(x)

                            SilhouetteScore = silhouette_score(X2_PCA, kmeans.labels_)
                            st.success(SilhouetteScore)

                            st.subheader('Davies Boundies Score:')
                            st.write(
                                'Essa métrica mede o equílibrio entre **quão distantes os clusters estão uns dos outros**')
                            st.write(' E o quão próximo os pontos dentro de cada cluster estão de seu centróide')
                            st.write('**Separação**: Quão distantes estão os centróides de centróides adjacentes')
                            st.write(
                                '**Compactação**: Mede o quão **espalhados** ou **compactos** estão os pontos dentro de um cluster')
                            st.write(
                                'Quanto menor a dispersão dos pontos dentro de um cluster (mais próximo do centróide), então melhor a compactação ')
                            st.write(
                                '**Valor zero**: Seria o melhor valor (ideal). Indica que os clusters estão perfeitamente compactados e seprados')
                            st.write('O valor ideal é raramente encontrado ')
                            st.write(
                                '**Valores entre 0 e 1**: São valores muito bons: Significa que os clusters estão separados e compactados')
                            st.write('**Valores acima de 1**: Clusters estão mais sobrepostos ou dispersos  ')
                            st.write("Quanto mais distantes de zero, então **menor a qualidade do agrupamento** ")
                            davies_bouldin = davies_bouldin_score(X2_PCA, kmeans.labels_)
                            st.success(davies_bouldin)


                    elif number_pca == 3:
                        pca = PCA(n_components=number_pca)
                        X_PCA = pca.fit_transform(x)
                        data5 = pd.DataFrame(X_PCA, columns=['PC1', 'PC2', 'PC3'])  # coloquei as PC's

                        data5['Cluster'] = kmeans.labels_  # Coloquei os clusters
                        data5 = data5.join(data[options])  # coloquei as variáveis categóricas

                        #Objetiva achar as variâncias dos centróides
                        data6 = data5.drop(columns=options)
                        dataN = data6  # tirei a variável categórica
                        data6['variancia_tot'] = data6['PC1'] + data6['PC2'] + data6['PC3']  # someis as variãncias de PCs
                        # st.dataframe(data6)
                        data7 = data6.drop(columns=['PC1', 'PC2', 'PC3'])  # Tirei as PC's

                        cluster_var = data7.groupby('Cluster')['variancia_tot'].std()

                        # Média de PC1 e PC2 em função dos clusters: acha-se  as coordenadas dos centróides(os centróides)
                        dataclust2 = dataN.groupby('Cluster').mean()  # são as coordenads  de centróides

                        array_cluster = dataclust2.values
                        st.dataframe(dataclust2)



                        st.dataframe(data5)  # Montei a tabela

                        # consigo fazer os gráfico dos meus clusters:
                        st.header('Kmeans Clustering')
                        categorica = st.selectbox('Escolha a  variável categórica que você quer ver no gráfico ',
                                                  ['Selecione'] + options, key='c1')






                        if categorica !='Selecione':
                            fig6 = px.scatter_3d(
                                data5,
                                x='PC1',
                                y='PC2',
                                z='PC3',
                                color=categorica,
                                opacity=0.7,
                                #symbol=categorica,
                                size_max=10,
                                title=f'PCA: PC1 vs PC2 vs PC3',

                            )
                            # Adiciona "círculos" em 3D
                            for cluster_label, row in dataclust2.iterrows():
                                x0 = row['PC1']
                                y0 = row['PC2']
                                z0 = row['PC3']

                                # Obtém seu raio (já calculado em cluster_var)
                                raio = cluster_var.loc[cluster_label] * 2

                                # Gera círculo no plano XY fixo em z0
                                theta = np.linspace(0, 2 * np.pi, 100)
                                x_circ = x0 + raio * np.cos(theta)
                                y_circ = y0 + raio * np.sin(theta)
                                z_circ = np.full_like(theta, z0)

                                # adiciona os centróides
                                fig6.add_trace(go.Scatter3d(
                                    x=dataclust2['PC1'],
                                    y=dataclust2['PC2'],
                                    z=dataclust2['PC3'],
                                    mode='markers',
                                    marker=dict(size=2, color='black'),
                                    showlegend=False
                                ))

                                # Adiciona trace do círculo
                                fig6.add_trace(go.Scatter3d(
                                    x=x_circ,
                                    y=y_circ,
                                    z=z_circ,
                                    mode='lines',
                                    line=dict(color='rgba(0, 0, 0, 0.5)', width=3),
                                    showlegend=False
                                ))





                            # Eixos com porcentagem (se quiser ajustar manualmente os nomes)
                            fig6.update_layout(
                                scene=dict(
                                    xaxis_title=f'Componente principal 1 ({pc1}%)',
                                    yaxis_title=f'Componente principal 2 ({pc2}%)',
                                    zaxis_title=f'Componente principal 3 ({pc3}%)',
                                    bgcolor='white'
                                ),
                                legend_title=categorica,
                                title_font_size=18
                            )

                            for i in range(varia1):
                                fig6.add_trace(go.Scatter3d(
                                    x=[0, vetores[i, 0]],
                                    y=[0, vetores[i, 1]],
                                    z=[0, vetores[i, 2]],
                                    mode='lines+markers+text',
                                    text=[None, name[i]],
                                    textposition='bottom center',  # 'middle right'  # ou 'top right', 'bottom center'
                                    line=dict(width=4, color='black'),
                                    showlegend=False
                                ))

                            # Mostra no Streamlit
                            st.plotly_chart(fig6, use_container_width=True)# gra´fic usa toda a largyra do app, quando True
                            st.dataframe(data5[categorica].value_counts())
                            st.header('Métricas para os resultados de clustering ')
                            st.subheader('Silhouette score')

                            st.write('Os resultado de **silhouette score** variam de 1 a -1 ')
                            st.write('Se próximos de 1, então os clusters estão bem **separados** e **compactados** ')
                            st.write( 'Se próximo de zero, então os clusters podem estar **sobrepostos** ')
                            st.write( 'Se negativos, então os pontos estão no **local errado** ')
                            pca = PCA(n_components=number_pca)
                            X2_PCA = pca.fit_transform(x)

                            SilhouetteScore = silhouette_score(X2_PCA, kmeans.labels_)
                            st.success(SilhouetteScore)

                            st.subheader('Davies Boundies Score:')
                            st.write('Essa métrica mede o equílibrio entre **quão distantes os clusters estão uns dos outros**')
                            st.write ( ' E o quão próximo os pontos dentro de cada cluster estão de seu centróide')
                            st.write('**Separação**: Quão distantes estão os centróides de centróides adjacentes')
                            st.write('**Compactação**: Mede o quão **espalhados** ou **compactos** estão os pontos dentro de um cluster' )
                            st.write( 'Quanto menor a dispersão dos pontos dentro de um cluster (mais próximo do centróide), então melhor a compactação ')
                            st.write('**Valor zero**: Seria o melhor valor (ideal). Indica que os clusters estão perfeitamente compactados e seprados')
                            st.write( 'O valor ideal é raramente encontrado ')
                            st.write( '**Valores entre 0 e 1**: São valores muito bons: Significa que os clusters estão separados e compactados')
                            st.write('**Valores acima de 1**: Clusters estão mais sobrepostos ou dispersos  ')
                            st.write("Quanto mais distantes de zero, então **menor a qualidade do agrupamento** ")
                            davies_bouldin = davies_bouldin_score(X2_PCA, kmeans.labels_)
                            st.success(davies_bouldin)

                    #Análise hierárquica de clustering
                    with tab3:
                        st.header('Análise hierárquica de clustering')
                        st.success('Dados normalizados')

                        data_clust2 = x.drop(columns = 'clustering')
                        data_clust3 = data_clust2.join(data[options])

                        st.dataframe(data_clust3)
                        lista2 = ['single', 'complete', 'ward', 'average']

                        lincar = st.selectbox('Escolha o linkage: ',
                                                  ['Selecione'] + lista2, key='c2')

                        if lincar != 'Selecione':
                            len_options = len(options)
                            #st.write(len_options)
                            if len_options == 1:

                                z = linkage(data_clust2, method=lincar)
                                st.dataframe(z)

                                #gráfico:
                                fig1, ax = plt.subplots(figsize = (10,6))
                                dendrogram(z, labels=data_clust3[options].values.ravel(), leaf_rotation=90, leaf_font_size=10)


                                ax.set_xlabel(f'{options}')
                                ax.set_ylabel("Distance")
                                ax.set_title(f'Hierarchical Clustering Dendrogram ({lincar})')
                                st.pyplot(fig1)
                                # Para fazer o gráfico:

                                clustering = st.number_input(
                                    'Escolha o número de clusters de acordo com o dendograma hierárquico:', min_value=2,
                                    max_value=n_linhas,
                                    placeholder='Digite a partir de 1', key='clu3')
                                st.success(f'Clustering: {clustering}')

                                clustering_1 = fcluster(z, clustering, criterion='maxclust')
                                # transformei em variável global para que possa ser usado nos scripts abaixo
                                if number_pca == 2:
                                    pca = PCA(n_components=number_pca)
                                    X_PCA = pca.fit_transform(x)
                                    data5 = pd.DataFrame(X_PCA, columns=['PC1', 'PC2'])  # coloquei as PC's

                                    data5['Cluster'] = clustering_1  # Coloquei os clusters
                                    data5 = data5.join(data[options])  # coloquei as variáveis categóricas
                                    # st.dataframe(data5)# Montei a tabela

                                    # variância dos clusters, o cícrculo de variância
                                    data6 = data5.drop(columns=options)
                                    dataN = data6  # tirei a variável categórica
                                    data6['variancia_tot'] = data6['PC1'] + data6['PC2']  # someis as variãncias de PCs
                                    # st.dataframe(data6)
                                    data7 = data6.drop(columns=['PC1', 'PC2'])  # Tirei as PC's

                                    cluster_var = data7.groupby('Cluster')['variancia_tot'].std()

                                    # Média de PC1 e PC2 em função dos clusters: acha-se  as coordenadas dos centróides(os centróides)
                                    dataclust2 = dataN.groupby('Cluster').mean()  # são as coordenads  de centróides

                                    array_cluster = dataclust2.values
                                    #st.dataframe(dataclust2)

                                    # consigo fazer os gráfico dos meus clusters:
                                    st.header('Kmeans Clustering')
                                    nome = data5.columns
                                    st.write(categorica)


                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    sns.scatterplot(
                                        data=data5,
                                        x='PC1',
                                        y='PC2',
                                        hue= categorica,  # Se `options` tiver só uma coluna
                                        palette='Set1',
                                        ax=ax
                                        )
                                    ax.set_xlabel(f'principal componet 1 ({pc1}%) ')
                                    ax.set_ylabel(f'principalcomponent 2 ({pc2}%)')

                                    for linha in range(len(array_cluster)):
                                        x_coord = array_cluster[linha][0]
                                        y_coord = array_cluster[linha][1]
                                        raio = cluster_var.iloc[linha] * 2  # Obtém o raio do cluster

                                        circle2 = plt.Circle((x_coord, y_coord), raio, color='#f5eef8', fill=True,
                                                             linestyle='solid', alpha=0.5)
                                        ax.add_patch(circle2)
                                        # Marcador do centróide
                                        ax.plot(x_coord, y_coord, 'o', color='black', markersize=2)

                                    # Anotações (nome das variáveis)
                                    texts = []
                                    for i, nome_variavel in enumerate(loading2.columns):
                                        x_loading = loading2.iloc[0, i] * escala
                                        y_loading = loading2.iloc[1, i] * escala

                                        text = ax.annotate(
                                            nome_variavel,
                                            (x_loading, y_loading),
                                            color="black",
                                            fontsize=7,
                                            va='bottom',
                                            rotation=45
                                        )

                                        texts.append(text)

                                    # Ajusta todos os textos juntos
                                    adjust_text(texts)

                                    # Setas
                                    for i, text in enumerate(texts):
                                        x_pos, y_pos = text.get_position()

                                        ax.arrow(
                                            0, 0,
                                            x_pos, y_pos,
                                            color="#873600",
                                            width=0.01,
                                            head_width=0.05,
                                            shape='full',
                                            length_includes_head=True
                                        )






                                    st.pyplot(fig)
                                    st.dataframe(data5[categorica].value_counts())

                                elif number_pca == 3:
                                    pca = PCA(n_components=number_pca)
                                    X_PCA = pca.fit_transform(x)
                                    data5 = pd.DataFrame(X_PCA, columns=['PC1', 'PC2', 'PC3'])  # coloquei as PC's

                                    data5['Cluster'] = clustering_1  # Coloquei os clusters  # Coloquei os clusters
                                    data5 = data5.join(data[options])  # coloquei as variáveis categóricas

                                    # Objetiva achar as variâncias dos centróides
                                    data6 = data5.drop(columns=options)
                                    dataN = data6  # tirei a variável categórica
                                    data6['variancia_tot'] = data6['PC1'] + data6['PC2'] + data6[
                                        'PC3']  # someis as variãncias de PCs
                                    # st.dataframe(data6)
                                    data7 = data6.drop(columns=['PC1', 'PC2', 'PC3'])  # Tirei as PC's

                                    cluster_var = data7.groupby('Cluster')['variancia_tot'].std()

                                    # Média de PC1 e PC2 em função dos clusters: acha-se  as coordenadas dos centróides(os centróides)
                                    dataclust2 = dataN.groupby('Cluster').mean()  # são as coordenads  de centróides

                                    array_cluster = dataclust2.values
                                    #st.dataframe(dataclust2)
                                    fig6 = px.scatter_3d(
                                        data5,
                                        x='PC1',
                                        y='PC2',
                                        z='PC3',
                                        color=categorica,
                                        opacity=0.7,
                                        # symbol=categorica,
                                        size_max=10,
                                        title=f'PCA: PC1 vs PC2 vs PC3',

                                    )
                                    # Adiciona "círculos" em 3D
                                    for cluster_label, row in dataclust2.iterrows():
                                        x0 = row['PC1']
                                        y0 = row['PC2']
                                        z0 = row['PC3']

                                        # Obtém seu raio (já calculado em cluster_var)
                                        raio = cluster_var.loc[cluster_label] * 2

                                        # Gera círculo no plano XY fixo em z0
                                        theta = np.linspace(0, 2 * np.pi, 100)
                                        x_circ = x0 + raio * np.cos(theta)
                                        y_circ = y0 + raio * np.sin(theta)
                                        z_circ = np.full_like(theta, z0)

                                        # adiciona os centróides
                                        fig6.add_trace(go.Scatter3d(
                                            x=dataclust2['PC1'],
                                            y=dataclust2['PC2'],
                                            z=dataclust2['PC3'],
                                            mode='markers',
                                            marker=dict(size=2, color='black'),
                                            showlegend=False
                                        ))

                                        # Adiciona trace do círculo
                                        fig6.add_trace(go.Scatter3d(
                                            x=x_circ,
                                            y=y_circ,
                                            z=z_circ,
                                            mode='lines',
                                            line=dict(color='rgba(0, 0, 0, 0.5)', width=3),
                                            showlegend=False
                                        ))

                                    # Eixos com porcentagem (se quiser ajustar manualmente os nomes)
                                    fig6.update_layout(
                                        scene=dict(
                                            xaxis_title=f'Componente principal 1 ({pc1}%)',
                                            yaxis_title=f'Componente principal 2 ({pc2}%)',
                                            zaxis_title=f'Componente principal 3 ({pc3}%)',
                                            bgcolor='white'
                                        ),
                                        legend_title=categorica,
                                        title_font_size=18
                                    )

                                    for i in range(varia1):
                                        fig6.add_trace(go.Scatter3d(
                                            x=[0, vetores[i, 0]],
                                            y=[0, vetores[i, 1]],
                                            z=[0, vetores[i, 2]],
                                            mode='lines+markers+text',
                                            text=[None, name[i]],
                                            textposition='bottom center',
                                            # 'middle right'  # ou 'top right', 'bottom center'
                                            line=dict(width=4, color='black'),
                                            showlegend=False
                                        ))

                                    # Mostra no Streamlit
                                    st.plotly_chart(fig6,
                                                    use_container_width=True)  # gra´fic usa toda a largyra do app, quando True









                            else:
                                options2 =  st.selectbox('Escolha o grupo para o gráfico ',
                                                  ['Selecione'] + options, key='c7')
                                if options2 != 'Selecione':
                                    z = linkage(data_clust2, method=lincar)
                                    st.dataframe(z)

                                    # gráfico:
                                    fig1, ax = plt.subplots(figsize=(10, 6))
                                    dendrogram(z, labels=data_clust3[options2].values.ravel(), leaf_rotation=90,
                                               leaf_font_size=10)

                                    ax.set_xlabel(f'{options2}')
                                    ax.set_ylabel("Distance")
                                    ax.set_title(f'Hierarchical Clustering Dendrogram ({lincar})')
                                    st.pyplot(fig1)
                                    # Para fazer o gráfico:

                                    clustering = st.number_input(
                                        'Escolha o número de clusters de acordo com o dendograma hierárquico:', min_value=2,
                                        max_value=n_linhas,
                                        placeholder='Digite a partir de 1', key='clu3')
                                    st.success(f'Clustering: {clustering}')
                                    clustering_1 = fcluster(z, clustering, criterion='maxclust')
                                      # transformei em variável global para que possa ser usado nos scripts abaixo

                                    if number_pca == 2:
                                        pca = PCA(n_components=number_pca)
                                        X_PCA = pca.fit_transform(x)
                                        data5 = pd.DataFrame(X_PCA, columns=['PC1', 'PC2'])  # coloquei as PC's

                                        data5['Cluster'] = clustering_1  # Coloquei os clusters
                                        data5 = data5.join(data[options])  # coloquei as variáveis categóricas
                                        # st.dataframe(data5)# Montei a tabela

                                        # variância dos clusters, o cícrculo de variância
                                        data6 = data5.drop(columns=options)
                                        dataN = data6  # tirei a variável categórica
                                        data6['variancia_tot'] = data6['PC1'] + data6['PC2']  # someis as variãncias de PCs
                                        # st.dataframe(data6)
                                        data7 = data6.drop(columns=['PC1', 'PC2'])  # Tirei as PC's

                                        cluster_var = data7.groupby('Cluster')['variancia_tot'].std()

                                        # Média de PC1 e PC2 em função dos clusters: acha-se  as coordenadas dos centróides(os centróides)
                                        dataclust2 = dataN.groupby('Cluster').mean()  # são as coordenads  de centróides

                                        array_cluster = dataclust2.values
                                        # st.dataframe(dataclust2)

                                        # consigo fazer os gráfico dos meus clusters:
                                        st.header(f'Hierarchical Clustering Dendrogram ({lincar})')
                                        nome = data5.columns
                                        #st.write(categorica)

                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        sns.scatterplot(
                                            data=data5,
                                            x='PC1',
                                            y='PC2',
                                            hue=options2,  # Se `options` tiver só uma coluna
                                            palette='Set1',
                                            ax=ax
                                        )
                                        ax.set_xlabel(f'principal componet 1 ({pc1}%) ')
                                        ax.set_ylabel(f'principalcomponent 2 ({pc2}%)')

                                        for linha in range(len(array_cluster)):
                                            x_coord = array_cluster[linha][0]
                                            y_coord = array_cluster[linha][1]
                                            raio = cluster_var.iloc[linha] * 2  # Obtém o raio do cluster

                                            circle2 = plt.Circle((x_coord, y_coord), raio, color='#f5eef8', fill=True,
                                                                 linestyle='solid', alpha=0.5)
                                            ax.add_patch(circle2)
                                            # Marcador do centróide
                                            ax.plot(x_coord, y_coord, 'o', color='black', markersize=2)

                                        # Anotações (nome das variáveis)
                                        texts = []
                                        for i, nome_variavel in enumerate(loading2.columns):
                                            x_loading = loading2.iloc[0, i] * escala
                                            y_loading = loading2.iloc[1, i] * escala

                                            text = ax.annotate(
                                                nome_variavel,
                                                (x_loading, y_loading),
                                                color="black",
                                                fontsize=7,
                                                va='bottom',
                                                rotation=45
                                            )

                                            texts.append(text)

                                        # Ajusta todos os textos juntos
                                        adjust_text(texts)

                                        # Setas
                                        for i, text in enumerate(texts):
                                            x_pos, y_pos = text.get_position()

                                            ax.arrow(
                                                0, 0,
                                                x_pos, y_pos,
                                                color="#873600",
                                                width=0.01,
                                                head_width=0.05,
                                                shape='full',
                                                length_includes_head=True
                                            )

                                        st.pyplot(fig)
                                        st.dataframe(data5[categorica].value_counts())

                                    elif number_pca == 3:
                                        pca = PCA(n_components=number_pca)
                                        X_PCA = pca.fit_transform(x)
                                        data5 = pd.DataFrame(X_PCA, columns=['PC1', 'PC2', 'PC3'])  # coloquei as PC's

                                        data5['Cluster'] = clustering_1  # Coloquei os clusters  # Coloquei os clusters
                                        data5 = data5.join(data[options])  # coloquei as variáveis categóricas

                                        # Objetiva achar as variâncias dos centróides
                                        data6 = data5.drop(columns=options)
                                        dataN = data6  # tirei a variável categórica
                                        data6['variancia_tot'] = data6['PC1'] + data6['PC2'] + data6[
                                            'PC3']  # someis as variãncias de PCs
                                        # st.dataframe(data6)
                                        data7 = data6.drop(columns=['PC1', 'PC2', 'PC3'])  # Tirei as PC's

                                        cluster_var = data7.groupby('Cluster')['variancia_tot'].std()

                                        # Média de PC1 e PC2 em função dos clusters: acha-se  as coordenadas dos centróides(os centróides)
                                        dataclust2 = dataN.groupby('Cluster').mean()  # são as coordenads  de centróides

                                        array_cluster = dataclust2.values
                                        # st.dataframe(dataclust2)
                                        fig6 = px.scatter_3d(
                                            data5,
                                            x='PC1',
                                            y='PC2',
                                            z='PC3',
                                            color=options2,
                                            opacity=0.7,
                                            # symbol=categorica,
                                            size_max=10,
                                            title=f'PCA: PC1 vs PC2 vs PC3',

                                        )
                                        # Adiciona "círculos" em 3D
                                        for cluster_label, row in dataclust2.iterrows():
                                            x0 = row['PC1']
                                            y0 = row['PC2']
                                            z0 = row['PC3']

                                            # Obtém seu raio (já calculado em cluster_var)
                                            raio = cluster_var.loc[cluster_label] * 2

                                            # Gera círculo no plano XY fixo em z0
                                            theta = np.linspace(0, 2 * np.pi, 100)
                                            x_circ = x0 + raio * np.cos(theta)
                                            y_circ = y0 + raio * np.sin(theta)
                                            z_circ = np.full_like(theta, z0)

                                            # adiciona os centróides
                                            fig6.add_trace(go.Scatter3d(
                                                x=dataclust2['PC1'],
                                                y=dataclust2['PC2'],
                                                z=dataclust2['PC3'],
                                                mode='markers',
                                                marker=dict(size=2, color='black'),
                                                showlegend=False
                                            ))

                                            # Adiciona trace do círculo
                                            fig6.add_trace(go.Scatter3d(
                                                x=x_circ,
                                                y=y_circ,
                                                z=z_circ,
                                                mode='lines',
                                                line=dict(color='rgba(0, 0, 0, 0.5)', width=3),
                                                showlegend=False
                                            ))

                                        # Eixos com porcentagem (se quiser ajustar manualmente os nomes)
                                        fig6.update_layout(
                                            scene=dict(
                                                xaxis_title=f'Componente principal 1 ({pc1}%)',
                                                yaxis_title=f'Componente principal 2 ({pc2}%)',
                                                zaxis_title=f'Componente principal 3 ({pc3}%)',
                                                bgcolor='white'
                                            ),
                                            legend_title=options2,
                                            title_font_size=18
                                        )

                                        for i in range(varia1):
                                            fig6.add_trace(go.Scatter3d(
                                                x=[0, vetores[i, 0]],
                                                y=[0, vetores[i, 1]],
                                                z=[0, vetores[i, 2]],
                                                mode='lines+markers+text',
                                                text=[None, name[i]],
                                                textposition='bottom center',
                                                # 'middle right'  # ou 'top right', 'bottom center'
                                                line=dict(width=4, color='black'),
                                                showlegend=False
                                            ))

                                        # Mostra no Streamlit
                                        st.plotly_chart(fig6,
                                                        use_container_width=True)  # gra´fic usa toda a largyra do app, quando True

                        with tab4:
                            st.header('Dendograma hierárquico ')
                            #Variável categórica
                            categorica2 = st.selectbox('Escolha a  variável categórica que você quer ver no dendograma ',
                                                      ['Selecione'] + options, key='c9')
                            lista_4= [
                                "euclidean",  # Distância Euclidiana (reta entre dois pontos)
                                "cityblock",  # Distância de Manhattan (soma das diferenças absolutas)
                                "minkowski",  # Generaliza Euclidiana e Manhattan com um parâmetro p
                                "chebyshev",  # Distância máxima em qualquer dimensão
                                "cosine",  # Mede o ângulo entre vetores (similaridade de cosseno)
                                "correlation",  # Mede dissimilaridade entre vetores normalizados
                                "canberra",  # Sensível a pequenas variações, útil para contagens
                                "braycurtis",  # Mede dissimilaridade baseada em proporções
                                "hamming",  # Mede a fração de elementos diferentes (para dados binários)
                                "jaccard" ] # Mede dissimilaridade entre conjuntos binários]
                                #Distância no gráfico:
                            distancia_1 = st.selectbox('Escolha a distância do dendograma hierárquico ',
                                                       ['Selecione'] + lista_4, key='c19')

                            lista3 = ['single', 'complete', 'ward', 'average']

                            lincar = st.selectbox('Escolha o linkage: ',
                                                  ['Selecione'] + lista3, key='C2')
                            if categorica2 != 'Selecione':
                                if distancia_1 != 'Selecione':
                                    st.dataframe(x)
                                    for pc_name in propor_autova3.index[:number_pca]:
                                        # Ordena todas as variáveis dessa PC em ordem decrescente (valor absoluto)
                                        loadings_sorted = propor_autova3.loc[pc_name].abs().sort_values(ascending=False)

                                        # Se quiser mostrar os valores originais (positivos ou negativos), pode usar:
                                        loadings_sorted_signed1 = propor_autova3.loc[pc_name].loc[loadings_sorted.index]
                                        # inserindo as 40 variáveis com maiores loadings
                                        loadings_sorted_signed = loadings_sorted_signed1.iloc[0:40]

                                        st.write(
                                            f" Dendograma das variáveis dos 40 primeiros Loadings  ordenados de forma decrescente  para {pc_name}:")
                                        # Converte o Index em DataFrame com uma coluna chamada 'Variáveis'
                                        df_variaveis = pd.DataFrame(loadings_sorted_signed.index, columns=["Variáveis"])

                                        dendo_gram = x[df_variaveis["Variáveis"]]
                                        dendo_gram[categorica2] = data[categorica2]
                                        # st.write(dendo_gram)
                                        n_colunas2 = dendo_gram.shape[1]
                                        # st.write(n_colunas2)
                                        # Média do grupo
                                        df_grouped = dendo_gram.groupby(categorica2).mean()
                                        st.write(df_grouped)
                                        # gráfico


                                        g = sns.clustermap(df_grouped, metric=distancia_1, method=lincar,
                                                           cmap='viridis', figsize=(15, 12))
                                        # Aumenta os rótulos dos eixos X e Y
                                        g.ax_heatmap.tick_params(axis='x', labelsize=14)
                                        g.ax_heatmap.tick_params(axis='y', labelsize=14)

                                        # Importante: pegar a figura (fig) do objeto retornado
                                        st.pyplot(g)  # Aqui você passa apenas g.fig, que é a Figure real
                                        #sns.set_theme()  # 🔧 Restaura o estilo padrão para os próximos gráficos
        with tab5:


            escolha1 = st.radio("Você deseja ver seus dados ?", ["Sim", "Não"], key = '99-N')
            if escolha1 == 'Sim':
                st.dataframe(data)
            data1 = data.to_dict()
            chaves = data1.keys()
            chaves1 = list(chaves)
            options = st.multiselect('Selecione todas as  variáveis categóricas ', ['Selecione'] + chaves1, key='m_2')
            if options:
                var1 = options
                # transformando esses dados em array numpy
                # print(escolha_categoria)
                # data1 = data.drop(columns = options) # estou conseguindo contar as NA's e retirar as Na's sem tirar a variável categórica
                data1 = data
                data2 = data1.isna().sum()
                data3 = data1

                # fazer uma função aqui! def retirarana(data na , data) e chamo novamente em outro lugar
                if data2.sum() == 0:
                    st.write(f'Você Não tem **NA** nas  variáveis de seus dados  ')
                    st.dataframe(data2)

                else:
                    st.write(f'Você  tem **NA** nas  variáveis de seus dados  ')
                    st.dataframe(data2)
                    st.write('Você gostaria de retira  as **NAs** ou substituir por valores médios?')
                    escolha_2 = st.radio("Você deseja ?", ["Substituir por Valores médios", "Retirar Na"],
                                         horizontal=True, key = 'm_1l')
                    if escolha_2 == "Substituir por Valores médios":
                        data3 = data3.fillna(data3.median(numeric_only=True))
                        st.dataframe(data3)
                        n_linhas = data3.shape[0]
                        n_colunas = data3.shape[1]
                        categorica = len(options)  # Número de variáveis categóricas
                        # categoricas = data.shape[1]
                        continua = n_colunas - categorica
                        st.success(f'Número de observações: {n_linhas}')
                        st.success(f'Número de variáveis: {n_colunas}')
                        st.success(f'Número de variáveis categóricas:{categorica}')
                        st.success(f'Número de variáveis contínuas:{continua}')
                    else:
                        st.write('Retirados NAs')
                        data3 = data3.dropna(axis=1)
                        st.dataframe(data3)
                        n_linhas = data3.shape[0]
                        n_colunas = data3.shape[1]
                        categorica = len(options)  # Número de variáveis categóricas
                        # categoricas = data.shape[1]
                        continua = n_colunas - categorica  # manter o mes
                        st.success(f'Número de observações: {n_linhas}')
                        st.success(f'Número de variáveis: {n_colunas}')
                        st.success(f'Número de variáveis categóricas:{categorica}')
                        st.success(f'Número de variáveis contínuas:{continua}')

                    variavel = st.radio('Quantas variáveis categóricas você deseja analisar?', [1, 2], horizontal=True)
                    if variavel == 1:
                        var1 = options[0]# agora temos uma string e não mais uma lista
                        #st.dataframe(data3[options].columns)
                        data3 = data3.drop(columns = var1)
                        st.write(len(data3.columns))
                        variatot = len(data3.columns)
                        normais = []
                        for resposta_var in data3.columns[0:3]:  # data frame variatot
                            formula = f'{resposta_var}~{var1}'

                            # extração dos resíduos advindo do modelo:
                            model = smf.ols(formula, data=data3).fit()
                            residuos = model.resid  # usa isso
                            valores_ajustados = model.fittedvalues
                            # eu inseri o nome das variáveis em seus respectivos valores ajustados
                            valores_ajustados_df = residuos.to_frame(name=resposta_var)
                            # print(valores_ajustados_df)
                            # global valores_ajustados_copia
                            valores_ajustados_copia = valores_ajustados_df.copy()
                            # print(valores_ajustados_df)
                            # Agora eu vou inserir as variáveis categóricas
                            #valores_ajustados_df[options] = data.iloc[options]







































                                       


































































