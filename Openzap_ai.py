from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import streamlit as st
import urllib
from tqdm import tqdm

from functions import *
import re
import tempfile


st.set_page_config(page_title="Agente Mensageiro - SaaS", layout="wide")
st.markdown(""" <style>
        footer {visibility: hidden;}
        </style> """, unsafe_allow_html=True)



tab1, tab2 = st.tabs(["✉️🚀 Geração e Envio de Mensagens",
                      "🕵️‍♂️💌 Rastreio de Mensagens"])

with tab1:
    st.markdown("<h1 style='font-size:200%; text-align: center; color: black; padding: 10px 0px 0px 0px;'" +
                ">Geração de Mensagens com CHATGPT</h1>",
                unsafe_allow_html=True)
    st.markdown('---')

    uploaded_file = st.file_uploader("Escolha um arquivo com contato dos clientes", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Lógica para criar DataFrame com base no tipo de arquivo
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')  # 'openpyxl' é um engine para ler arquivos xlsx
        df = df.astype('str')
        with st.expander('Ver arquivo completo'):

            len_df = df.shape[0]
            st.write("Base de dados do arquivo: " + str(len_df) + " clientes na lista")

            st.dataframe(df, height=200)


    st.markdown('---')

    api_key = st.text_input('Insira sua API-Key da OPENAI')
    if len(api_key) == 0:
        st.warning(
            'Para gerar as mensagens elaboradas pelo ChatGPT-3.5 é necessário adicionar sua API-Key '
            'na caixa de texto acima. Caso ainda não tenha uma chave de API, '
            'você pode criá-la acessando o seguinte endereço: https://platform.openai.com/account/api-keys.',
            icon='🗝️')

    col1, col2 = st.columns([1, 1])
    caract = col1.slider('Selecione o n° de caracteres da mensagem gerada:', min_value=200, max_value=1000, value=400)
    temperature = col2.slider('Selecione a Criatividade do ChatGPT:', min_value=0.0, max_value=1.0, value=0.5)

    col1, col2 = st.columns([1, 1])
    with col1:
        cat_tolist = ['Visitante', 'Semanal', 'Quinzenal','Mensal', 'Semestral', 'Trimestral', 'Anual']
        rotina = st.multiselect("Filtre os clientes por sua Rotina de compra:",
                                      options=cat_tolist, default='Visitante', key=558)

    with col2:
        fidel_tolist = ['Novo','Recente', 'Aberto', 'Atrasado', 'Ausente']
        frequencia = st.multiselect("Filtre os Clientes por sua Classe de Consumo:",
                                        options=fidel_tolist, default='Novo', key=559)


    solicitado = st.text_area('')

    mensagem = st.button('Gerar mensagem com ChatGPT!')

    st.markdown('---')


    if mensagem and len(api_key) != 0:
        mensagem_ai = generate_summary(solicitado, temperature, api_key, rotina, frequencia, caract)
        st.markdown("<h3 style='font-size:100%; text-align: left; color: black; padding: 0px 0px 0px 0px;'" +
                    ">Mensagem Personalizada: "+str(len(mensagem_ai))+" caracteres</h3>", unsafe_allow_html=True)
        st.text('')
        st.markdown(mensagem_ai.strip())
        st.markdown('---')



    st.markdown("<h1 style='font-size:200%; text-align: center; color: black; padding: 0px 0px 0px 0px;'" +
                ">Envio Automático de Mensagens por WhatsApp</h1>",
                unsafe_allow_html=True)
    st.markdown('---')


    mensagem_zap = st.text_area('Copie e edite a mensagem para enviar por whatsapp:')


    if len(mensagem_zap) > 100:

        if uploaded_file is None:
            st.warning('Selecione um arquivo no começo da página!', icon="⚠️")

        else:
            df['Mensagem Whatsapp'] = mensagem_zap.strip()


            # Função para formatar o nome
            def formatar_nome(row):
                if row['Tipo de Pessoa'] == 'F':
                    return row['Nome do contato'].split()[0].capitalize().title()
                else:
                    return row['Nome do contato'].capitalize().title()


            # Aplica a função usando apply e uma função lambda
            df['Nome Formatado'] = df.apply(lambda row: formatar_nome(row), axis=1)

            palavras_indesejadas = ['De Sa', 'Eireli', 'Ltda', 'Me', 'Ltd']
            def remover_palavras_indesejadas(texto, palavras_indesejadas):
                padrao = r'\b(?:{})\b'.format('|'.join(map(re.escape, palavras_indesejadas)))
                return re.sub(padrao, '', texto)
            df['Nome Formatado'] = df['Nome Formatado'].apply(
                lambda x: remover_palavras_indesejadas(x, palavras_indesejadas))

            def remover_numeros(texto):
                texto_sem_numeros = re.sub(r'\d', '', texto)  # Remove todos os dígitos da string
                return texto_sem_numeros
            df['Nome Formatado'] = df['Nome Formatado'].apply(remover_numeros)


            def remover_especiais(texto):
                remover_especiais = texto.replace('.', '').replace(',', '')
                return remover_especiais
            df['Nome Formatado'] = df['Nome Formatado'].apply(remover_especiais)

            df['Nome Formatado'] = df['Nome Formatado'].str.strip()
            df['Mensagem Whatsapp'] = (df.apply(lambda row: row['Mensagem Whatsapp']
                                                .replace('Cliente', row['Nome Formatado']), axis=1))

            df = df[['Celular', 'e-mail' ,'Nome do contato', 'Mensagem Whatsapp', 'Nome Formatado', 'Tipo de Pessoa' ]]


            with st.expander('Fase de testes:'):
                novo_numero = st.text_input('Digite o novo número de celular:')

                if st.checkbox('Substituir Números'):
                    df['Celular'] = novo_numero

                st.markdown('---')

            st.markdown("<h2 style='font-size:125%; text-align: center; color: black; padding: 0px 0px 20px 0px;'" +
                        ">Base de dados - pré envio das mensagens: " + str(len(df)) + " mensagens no total</h2>",
                        unsafe_allow_html=True)

            df = st.experimental_data_editor(df, height=400, use_container_width=True)

            st.markdown('---')


            df_download = df.to_csv(index=False).encode('utf-8')
            data_e_horario_atual = datetime.now()
            data_e_horario_formatados = data_e_horario_atual.strftime("%Y-%m-%d_%H-%M")

            if st.download_button(label="Enviar mensagens por Whatsapp!", data=df_download,
                                  file_name=f"envio_mensagens_whatsapp_{data_e_horario_formatados}.csv",
                                  mime='csv', key=442):

                with st.expander('conferir mensagens'):
                    st.dataframe(df, height=200)
                    st.markdown('---')

                options = Options()

                def get_driver():
                    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

                navegador = get_driver()
                navegador.get("https://web.whatsapp.com/")

                # Barra de progresso
                progress_text = "Enviando mensagens..."
                my_bar = st.progress(0, text=progress_text)

                for linha in tqdm(df.index, desc="Progresso", leave=False):
                    # enviar uma mensagem para a pessoa
                    nome = df.loc[linha, "Nome do contato"]
                    mensagem = df.loc[linha, "Mensagem Whatsapp"]
                    telefone = df.loc[linha, "Celular"]

                    texto = mensagem.replace("fulano", nome)
                    texto = urllib.parse.quote(texto)

                    link = f"https://web.whatsapp.com/send?phone={telefone}&text={texto}"

                    navegador.get(link)

                    while len(navegador.find_elements(By.ID, 'side')) < 1:
                        time.sleep(2)
                    time.sleep(10)

                    try:
                        if len(navegador.find_elements(
                                By.XPATH,
                                '/html/body/div[1]/div/div/div[5]/div/footer/div[1]/div/span[2]/div/div[2]/div[2]')) != 0:
                            navegador.find_element(
                                By.XPATH,
                                '/html/body/div[1]/div/div/div[5]/div/footer/div[1]/div/span[2]/div/div[2]/div[2]').click()
                            time.sleep(5)

                            # Atualize a barra de progresso
                        progress_percent = (linha + 1) / len(df.index)

                        progress_percent_text = round(progress_percent * 100, 2)
                        my_bar.progress(progress_percent, text=str(progress_percent_text) + '%')
                        st.info('Mensagem enviada: ' + nome + ' | ' + telefone, icon='📩')


                    except:
                        st.error('Erro em enviar a mensagem', icon='🚨')

                navegador.quit()
                my_bar.progress(100, text='100%')
                st.success('Todas as mensagem enviada com sucesso!', icon='✅')



with tab2:

    df_up = st.file_uploader("Arquivo para Rastreio de Mensagens", type=["csv", "xlsx"])

    if df_up is not None:
        # Lógica para criar DataFrame com base no tipo de arquivo
        if df_up.name.endswith('.csv'):
            df = pd.read_csv(df_up)
        elif df_up.name.endswith('.xlsx'):
            df = pd.read_excel(df_up, engine='openpyxl')  # 'openpyxl' é um engine para ler arquivos xlsx
        df = df.astype('str')
        st.markdown('---')
        with st.expander('Ver arquivo antes do rastreio das mensagens'):

            len_df = df.shape[0]
            st.write("Base de dados do arquivo: " + str(len_df) + " clientes na lista")

            st.dataframe(df, height=200)

    elif df_up is None and len(mensagem_zap) > 100 and uploaded_file is not None:
        df = df
        st.markdown('---')
        with st.expander('Ver arquivo antes do rastreio das mensagens'):

            len_df = df.shape[0]
            st.write("Base de dados do arquivo: " + str(len_df) + " clientes na lista")

            st.dataframe(df, height=200)

    st.markdown('---')

    teste_rastreio = st.button('Rastreio de mensagens')

    if teste_rastreio:
        options = Options()


        def get_driver():
            return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


        navegador = get_driver()

        link = f"https://web.whatsapp.com"
        navegador.get(link)

        df['Respostas Whatsapp'] = [[] for _ in range(len(df))]
        df['Numero de Respostas'] = [0 for _ in range(len(df))]

        for linha in tqdm(df.index, desc="Progresso", leave=False):
            telefone = df.loc[linha, "Celular"]
            mensagem = df.loc[linha, "Mensagem Whatsapp"]

            texto = urllib.parse.quote(mensagem)

            link = f"https://web.whatsapp.com/send?phone={telefone}"
            navegador.get(link)

            while len(navegador.find_elements(By.ID, 'side')) < 1:
                time.sleep(3)
            time.sleep(5)

            elementos = navegador.find_elements(By.XPATH,
                                                '//*[@class="_11JPr selectable-text copyable-text"]/span')

            textos = []

            # Itere pelos elementos e obtenha o texto de cada um
            for elemento in elementos:
                texto = elemento.text.strip()
                textos.append(texto)

            posicao = textos.index(mensagem)
            respostas = []

            # Exibir os itens da lista a partir da posição encontrada
            for i in range(posicao + 1, len(textos)):
                respostas.append(textos[i])

            num_respostas = len(respostas)

            df.at[linha, 'Respostas Whatsapp'] = respostas
            df.at[linha, 'Numero de Respostas'] = num_respostas

            df.at[linha, 'Numero de Respostas'] = max(num_respostas, df.at[linha, 'Numero de Respostas'])

            df['Numero de Respostas'].fillna(0, inplace=True)

        # Classificar os clientes com base no número de respostas
        df['Status Resposta'] = df['Numero de Respostas'].apply(classificar_cliente)

        df = df[
            ['Status Resposta', 'Numero de Respostas', 'Nome do contato', 'Respostas Whatsapp', 'Mensagem Whatsapp']]

        st.session_state.df = df
        navegador.quit()

    # Inicializar a variável de sessão
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Exibir DataFrame no Streamlit
    if st.session_state.df is not None:

        df = st.session_state.df

        st.markdown('---')

        st.dataframe(df, use_container_width=True)

        st.markdown('---')

        container = st.container()
        col1, col2, col3 = container.columns([1, 1, 1])
        with col1:
            st.markdown("<h2 style='font-size:120%; text-align: center; color: black; padding: 10px 0px 10px 0px;'" +
                        ">Frequência do Status das Respostas</h2>", unsafe_allow_html=True)

            count = (df.groupby('Status Resposta').count()
                     .sort_values('Numero de Respostas', ascending=False).reset_index())

            fig = bar_plot_horiz(count.head(10), 'Status Resposta', 'Numero de Respostas', '#000080', 'Status Resposta')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<h2 style='font-size:120%; text-align: center; color: black; padding: 10px 0px 10px 0px;'" +
                        ">Total do Número Respostas e sua Meta</h2>", unsafe_allow_html=True)

            fig2 = metrica_respostas(df)
            st.plotly_chart(fig2, use_container_width=True)

        with col3:
            st.markdown("<h2 style='font-size:120%; text-align: center; color: black; padding: 10px 0px 10px 0px;'" +
                        ">Palavras mais usadas nas Respostas</h2>", unsafe_allow_html=True)

            try:
                fig3 = wordcloud(df)
                st.pyplot(fig3)
            except:
                st.error('Sem Respostas no Whatsapp')








st.markdown('---')