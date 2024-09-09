import streamlit as st
import os
import shutil
import zipfile

# Definisci la directory dove saranno salvati i file
FILE_DIRECTORY = 'file_library'
os.makedirs(FILE_DIRECTORY, exist_ok=True)

st.title("Libreria Virtuale - Caricamento di Cartelle ed Eliminazione File/Cartelle")

# Funzione per visualizzare la lista dei file e cartelle nella directory
def list_files_and_folders(directory):
    items = os.listdir(directory)
    return items

# Funzione per eliminare file
def delete_file(file_name):
    file_path = os.path.join(FILE_DIRECTORY, file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)
        st.success(f"File '{file_name}' eliminato con successo!")
    else:
        st.error(f"'{file_name}' non è un file valido!")

# Funzione per eliminare cartella e il suo contenuto
def delete_folder(folder_name):
    folder_path = os.path.join(FILE_DIRECTORY, folder_name)
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        st.success(f"Cartella '{folder_name}' eliminata con successo!")
    else:
        st.error(f"'{folder_name}' non è una cartella valida!")

# Caricamento file ZIP
uploaded_zip = st.file_uploader("Carica una cartella compressa (.zip) nella libreria virtuale", type=["zip"])

# Se il file ZIP è stato caricato, estrai tutto nella directory
if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(FILE_DIRECTORY)
    st.success(f"Cartella '{uploaded_zip.name}' estratta con successo nella libreria virtuale!")

# Elimina file o cartella
st.subheader("Elimina file o cartella dalla libreria")
existing_items = list_files_and_folders(FILE_DIRECTORY)

if existing_items:
    item_to_delete = st.selectbox("Seleziona un file o una cartella da eliminare", existing_items)
    
    if st.button("Elimina"):
        item_path = os.path.join(FILE_DIRECTORY, item_to_delete)
        
        if os.path.isfile(item_path):
            delete_file(item_to_delete)
        elif os.path.isdir(item_path):
            delete_folder(item_to_delete)
        
        # Aggiorna manualmente la lista degli elementi eliminati
        existing_items = list_files_and_folders(FILE_DIRECTORY)
        if existing_items:
            st.selectbox("Aggiornato: seleziona un altro file o cartella da eliminare", existing_items)
else:
    st.info("Nessun file o cartella disponibile per l'eliminazione.")

