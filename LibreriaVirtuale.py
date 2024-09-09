import streamlit as st
import os

# Definisci la directory per la libreria virtuale
FILE_DIRECTORY = 'footballquant_library'

# Funzione per verificare e creare la directory se non esiste
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            st.write(f"Directory '{directory}' creata con successo.")
        except Exception as e:
            st.error(f"Errore nella creazione della directory '{directory}': {e}")

# Assicurati che la directory esista
ensure_directory_exists(FILE_DIRECTORY)

st.title("Gestione dei File nella Libreria Virtuale")

# Funzione per visualizzare la lista dei file nella directory
def list_files(directory):
    try:
        if os.path.exists(directory):
            return [f for f in os.listdir(directory) if f.endswith('.xlsx')]
        else:
            st.error(f"La directory '{directory}' non esiste.")
            return []
    except Exception as e:
        st.error(f"Errore nella lettura della directory '{directory}': {e}")
        return []

# Mostra i file nella libreria virtuale
file_names = list_files(FILE_DIRECTORY)

if file_names:
    st.write("File disponibili:")
    st.write(file_names)
else:
    st.info("Nessun file .xlsx disponibile nella libreria virtuale.")

# Opzioni per eliminare file
st.subheader("Elimina un file dalla libreria")

if file_names:
    file_to_delete = st.selectbox("Seleziona un file da eliminare", file_names)
    
    if st.button("Elimina"):
        file_path = os.path.join(FILE_DIRECTORY, file_to_delete)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                st.success(f"File '{file_to_delete}' eliminato con successo!")
            else:
                st.error(f"Impossibile trovare il file '{file_to_delete}'.")
        except Exception as e:
            st.error(f"Errore nell'eliminazione del file '{file_to_delete}': {e}")
else:
    st.info("Nessun file disponibile per l'eliminazione.")
