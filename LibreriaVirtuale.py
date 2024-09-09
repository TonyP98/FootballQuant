import streamlit as st
import os
import shutil
import zipfile

# Definisci la directory per la libreria virtuale
FILE_DIRECTORY = '"C:\Users\Utente\file_library"'
os.makedirs(FILE_DIRECTORY, exist_ok=True)

st.title("Libreria Virtuale - Caricamento e Gestione di File")

# Funzione per visualizzare la lista dei file e cartelle nella directory
def list_files_and_folders(directory):
    items = os.listdir(directory)
    return items

# Funzione per caricare file ZIP
def upload_zip_file(uploaded_zip):
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(FILE_DIRECTORY)
    st.success(f"File ZIP '{uploaded_zip.name}' estratto con successo nella libreria virtuale!")

# Carica un file ZIP
uploaded_zip = st.file_uploader("Carica un file ZIP (.zip) nella libreria virtuale", type=["zip"])

if uploaded_zip:
    upload_zip_file(uploaded_zip)

# Mostra i file e le cartelle nella libreria
st.subheader("File e Cartelle nella Libreria Virtuale")
existing_items = list_files_and_folders(FILE_DIRECTORY)

if existing_items:
    st.write(existing_items)
else:
    st.info("Nessun file o cartella presente nella libreria.")

# Opzioni per eliminare file o cartelle
st.subheader("Elimina file o cartella dalla libreria")

if existing_items:
    item_to_delete = st.selectbox("Seleziona un file o una cartella da eliminare", existing_items)
    
    if st.button("Elimina"):
        item_path = os.path.join(FILE_DIRECTORY, item_to_delete)
        
        if os.path.isfile(item_path):
            os.remove(item_path)
            st.success(f"File '{item_to_delete}' eliminato con successo!")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            st.success(f"Cartella '{item_to_delete}' eliminata con successo!")
        
        # Ricarica la lista aggiornata degli elementi
        existing_items = list_files_and_folders(FILE_DIRECTORY)
        st.write(existing_items)
else:
    st.info("Nessun file o cartella disponibile per l'eliminazione.")
