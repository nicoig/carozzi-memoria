# Para crear el requirements.txt ejecutamos 
# pipreqs --encoding=utf8 --force

# Primera Carga a Github
# git init
# git add .
# git commit -m "primer commit"
# git remote add origin https://github.com/nicoig/carozzi-memoria.git
# git push -u origin master

# Actualizar Repo de Github
# git add .
# git commit -m "Se actualizan las variables de entorno"
# git push origin master

# Para eliminar un repo cargado
# git remote remove origin

# En Render
# agregar en variables de entorno
# PYTHON_VERSION = 3.9.12

###############################################################

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pickle
from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message
import io
import asyncio


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") 

async def main():

    async def storeDocEmbeds(file, filename):
    
        reader = PdfReader(file)
        corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])
        
        splitter =  RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,)
        chunks = splitter.split_text(corpus)
        
        embeddings = OpenAIEmbeddings(openai_api_key = api_key)
        vectors = FAISS.from_texts(chunks, embeddings)
        
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(vectors, f)

        
    async def getDocEmbeds(file, filename):
        
        if not os.path.isfile(filename + ".pkl"):
            await storeDocEmbeds(file, filename)
        
        with open(filename + ".pkl", "rb") as f:
            global vectors
            vectors = pickle.load(f)
            
        return vectors
    

    async def conversational_chat(query):
        result = qa({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")

    if 'history' not in st.session_state:
        st.session_state['history'] = []

        
    # Estableciendo la franja superior
    st.image("img/franja_inferior_1.png")

    # Agregar un espacio o salto
    st.write("")

    # Estableciendo el logo de Carozzi
    st.image("img/logo_carozzi_chat.png", width=300)

    # Estableciendo el título, subtítulo y descripción de Carozzi  
    #st.title("Carozzi Chat")


    # Título y descripción
    st.subheader("Descubre lo último sobre Carozzi")
    st.markdown("<p style='color: black; font-size: 15px;'>¡Hablemos de innovación, Transformación Digital, Medio Ambiente, Calidad, Salud y Nutrición, Marketing Responsable, Sostenibilidad y mucho más! Haz tus preguntas y descubre todo lo que Carozzi está haciendo para cambiar el mundo.</p>", unsafe_allow_html=True)


    if 'ready' not in st.session_state:
        st.session_state['ready'] = False

    # Aquí es donde cargamos el archivo PDF fijo en lugar del cargado por el usuario
    file_path = "Carozzi.pdf"
    with open(file_path, "rb") as f:
        file = f.read()

    with st.spinner("Procesando..."):
        vectors = await getDocEmbeds(io.BytesIO(file), file_path)
        qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=vectors.as_retriever(), return_source_documents=True)

    st.session_state['ready'] = True

    st.divider()

    if st.session_state['ready']:
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["¡Bienvenido! Realiza tu consulta"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hola!"]

        # container for chat history
        response_container = st.container()

        # container for text box
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Ingrese su solicitud:", placeholder="Ej: ¿Que planes tiene Carozzi en Transformación Digital?", key='input')
                submit_button = st.form_submit_button(label='Enviar')

            if submit_button and user_input:
                output = await conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")


if __name__ == "__main__":
    asyncio.run(main())
    st.image("img/franja_inferior_1.png", use_column_width=True)
