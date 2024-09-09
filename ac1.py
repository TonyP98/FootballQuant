import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Imposta la modalità della pagina a "wide"
st.set_page_config(layout="wide")

# Funzione per calcolare la regressione lineare
def linear_regression(X, y):
    X = sm.add_constant(X)  # Aggiungi una colonna di 1 per l'intercetta
    model = sm.OLS(y, X).fit()
    return model

# Funzione per calcolare i valori previsti
def calculate_forecast(df, period):
    df['Valori Previsti'] = np.nan

    for i in range(period - 1, len(df)):
        # Prendi i periodi precedenti
        y = df['Somma Cumulativa'][i - period + 1:i + 1]
        X = df['Prima Colonna'][i - period + 1:i + 1]

        # Calcola la regressione lineare
        model = linear_regression(X, y)
        intercept = model.params[0]  # Intercetta
        slope = model.params[1]      # Coefficiente angolare

        # Calcola il valore previsto
        if i < len(df):
            df.at[i, 'Regressione'] = slope
            df.at[i, 'Intercetta'] = intercept
            if i >= period - 1:
                df.at[i, 'Valori Previsti'] = df.at[i, 'Prima Colonna'] * df.at[i - 1, 'Regressione'] + df.at[i - 1, 'Intercetta']

    return df

# Funzione per calcolare le bande di deviazione standard
def calculate_bollinger_bands(diff):
    rolling_mean = pd.Series(diff).rolling(window=len(diff), min_periods=1).mean()
    rolling_std = pd.Series(diff).rolling(window=len(diff), min_periods=1).std()
    # Calcola bande per 1°, 2° e 3° deviazione standard
    bands = {
        'mean': rolling_mean,
        'upper_1std': rolling_mean + 1 * rolling_std,
        'lower_1std': rolling_mean - 1 * rolling_std,
        'upper_2std': rolling_mean + 2 * rolling_std,
        'lower_2std': rolling_mean - 2 * rolling_std,
        'upper_3std': rolling_mean + 3 * rolling_std,
        'lower_3std': rolling_mean - 3 * rolling_std
    }
    return bands

# Funzione per calcolare l'istogramma
def create_histogram(data, bin_size):
    bins = np.arange(data.min(), data.max() + bin_size, bin_size)
    hist, _ = np.histogram(data, bins=bins)
    return hist, bins

# Funzione per calcolare la percentuale di valori minori rispetto al successivo
def calculate_percentage_smaller(df, bin_edges):
    # Crea una colonna per indicare se il valore corrente è minore del successivo
    df['Minore Successivo'] = df['Differenza'] < df['Differenza'].shift(-1)
    
    # Crea una lista per memorizzare le percentuali
    percentages = []
    
    # Per ogni intervallo dell'istogramma
    for i in range(len(bin_edges) - 1):
        lower_edge = bin_edges[i]
        upper_edge = bin_edges[i + 1]
        
        # Filtra i valori di Differenza nell'intervallo corrente
        mask = (df['Differenza'] >= lower_edge) & (df['Differenza'] < upper_edge)
        interval_data = df[mask]
        
        # Calcola la percentuale di valori che sono minori del successivo
        if len(interval_data) > 0:
            percentage_smaller = interval_data['Minore Successivo'].mean() * 100
        else:
            percentage_smaller = 0
        
        percentages.append(percentage_smaller)
    
    return percentages

def count_terms_in_intervals(df):
    # Calcola le bande di deviazione standard
    bands = calculate_bollinger_bands(df['Differenza'].dropna())
    
    # Estrai i valori finali per ciascuna banda
    upper_3std = bands['upper_3std'].iloc[-1]
    lower_3std = bands['lower_3std'].iloc[-1]
    upper_2std = bands['upper_2std'].iloc[-1]
    lower_2std = bands['lower_2std'].iloc[-1]
    upper_1std = bands['upper_1std'].iloc[-1]
    lower_1std = bands['lower_1std'].iloc[-1]
    
    # Conta i valori di Differenza negli intervalli specificati
    counts = {
        'lower_3std_lower_2std': df[(df['Differenza'] >= lower_3std) & (df['Differenza'] <= lower_2std)].shape[0],
        'lower_2std_lower_1std': df[(df['Differenza'] >= lower_2std) & (df['Differenza'] <= lower_1std)].shape[0],
        'lower_1std_upper_1std': df[(df['Differenza'] >= lower_1std) & (df['Differenza'] <= upper_1std)].shape[0],
        'upper_1std_upper_2std': df[(df['Differenza'] > upper_1std) & (df['Differenza'] <= upper_2std)].shape[0],
        'upper_2std_upper_3std': df[(df['Differenza'] > upper_2std) & (df['Differenza'] <= upper_3std)].shape[0],
    }
    
    return counts

# Funzione per calcolare la percentuale rispetto al totale di ciascun intervallo
def calculate_percentage_of_total(counts):
    total = sum(counts.values())
    if total == 0:
        return {interval: 0 for interval in counts}
    percentages = {interval: (count / total) * 100 for interval, count in counts.items()}
    return percentages

def highlight_bin(fig_hist, fig_bar, bin_center, bin_edges, hist_color='blue', highlight_color='red'):
    # Trova il bin corretto in base ai bin_edges di fig.hist
    bin_index = None
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= bin_center < bin_edges[i + 1]:
            bin_index = i
            break

    # Se il bin è stato trovato, aggiorna entrambi i grafici
    if bin_index is not None:
        # Aggiorna fig.hist per evidenziare il bin con il colore rosso
        fig_hist.update_traces(marker=dict(color=[
            highlight_color if i == bin_index else hist_color for i in range(len(bin_edges) - 1)
        ]))

        # Aggiorna fig.bar per evidenziare il bin corrispondente con il colore rosso
        fig_bar.update_traces(marker=dict(color=[
            highlight_color if i == bin_index else hist_color for i in range(len(bin_edges) - 1)
        ]))

    return fig_hist, fig_bar


def count_consecutive_conditions(diff, bands, condition, opposite_condition):
    consecutive_counts = []
    count = 0

    # Applicare le bande alla Serie
    upper_1std = bands['upper_1std']
    lower_1std = bands['lower_1std']
    upper_2std = bands['upper_2std']
    lower_2std = bands['lower_2std']
    upper_3std = bands['upper_3std']
    lower_3std = bands['lower_3std']

    for i, value in enumerate(diff):
        # Verifica se il valore soddisfa la condizione
        if condition(value, upper_1std[i], lower_1std[i], upper_2std[i], lower_2std[i], upper_3std[i], lower_3std[i]):
            count += 1
        elif opposite_condition(value, upper_1std[i], lower_1std[i], upper_2std[i], lower_2std[i], upper_3std[i], lower_3std[i]):
            if count > 0:
                consecutive_counts.append(count)
            count = 0
        # Se nessuna condizione si verifica, continua a contare finché non si verifica la condizione opposta
        else:
            if count > 0:
                consecutive_counts.append(count)
            count = 0

    if count > 0:  # Per l'ultimo gruppo se non è stato chiuso
        consecutive_counts.append(count)

    return consecutive_counts

def condition1(value, upper_1std, lower_1std, upper_2std, lower_2std, upper_3std, lower_3std):
    return value < lower_2std

def opposite_condition1(value, upper_1std, lower_1std, upper_2std, lower_2std, upper_3std, lower_3std):
    return value >= lower_2std

def condition2(value, upper_1std, lower_1std, upper_2std, lower_2std, upper_3std, lower_3std):
    return lower_2std <= value < lower_1std

def opposite_condition2(value, upper_1std, lower_1std, upper_2std, lower_2std, upper_3std, lower_3std):
    return value >= lower_1std

def condition3(value, upper_1std, lower_1std, upper_2std, lower_2std, upper_3std, lower_3std):
    return upper_1std <= value < upper_2std

def opposite_condition3(value, upper_1std, lower_1std, upper_2std, lower_2std, upper_3std, lower_3std):
    return value <= upper_1std 

def condition4(value, upper_1std, lower_1std, upper_2std, lower_2std, upper_3std, lower_3std):
    return value >= upper_2std

def opposite_condition4(value, upper_1std, lower_1std, upper_2std, lower_2std, upper_3std, lower_3std):
    return value < upper_2std

def plot_consecutive_histogram(consecutive_counts, label, title):
    # Calcola la frequenza di ciascun numero di termini consecutivi
    if consecutive_counts:
        counts_frequency = [consecutive_counts.count(i) for i in range(1, max(consecutive_counts)+1)]
        total_counts = sum(counts_frequency)
        percent_frequency = [count / total_counts * 100 for count in counts_frequency]
    else:
        counts_frequency = []
        percent_frequency = []

    # Crea il grafico
    fig = go.Figure()
    if percent_frequency:
        fig.add_trace(go.Bar(
            x=list(range(1, len(percent_frequency) + 1)),  # Numero di termini consecutivi
            y=percent_frequency,  # Percentuale di ciascun numero di termini consecutivi
            marker_color='blue',
            name=label
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Numero di termini consecutivi',
        yaxis_title='Percentuale (%)',
        template='plotly_white',
        barmode='group'
    )

    return fig

# Carica la libreria virtuale
FILE_DIRECTORY = "C:\Users\Utente\file_library"
file_names = [f for f in os.listdir(FILE_DIRECTORY) if f.endswith('.xlsx')]

# Seleziona il file dalla libreria
selected_file = st.selectbox("Seleziona il file da visualizzare:", file_names)

# Aggiungi il menu a tendina per la selezione della condizione
condition = st.selectbox("Seleziona la condizione:", ["Win", "Draw", "Lose"])

if selected_file:
    file_path = os.path.join(FILE_DIRECTORY, selected_file)
    df = pd.read_excel(file_path)

    # Assicurati che ci siano almeno due colonne
    if df.shape[1] >= 2:
        # Usa la prima colonna come X e la seconda come B
        df['Prima Colonna'] = df.iloc[:, 0]
        df['Seconda Colonna'] = df.iloc[:, 1]

        # Modifica la funzione 'Vero/Falso' in base alla condizione selezionata
        if condition == "Win":
            df['Vero/Falso'] = df['Seconda Colonna'].apply(lambda x: "Vero" if "W" in str(x) else "Falso")
        elif condition == "Draw":
            df['Vero/Falso'] = df['Seconda Colonna'].apply(lambda x: "Vero" if "D" in str(x) else "Falso")
        elif condition == "Lose":
            df['Vero/Falso'] = df['Seconda Colonna'].apply(lambda x: "Vero" if "L" in str(x) else "Falso")

        # Conta i "Vero" e aggiungi la nuova colonna 'Conta Vero'
        df['Conta Vero'] = df['Vero/Falso'].apply(lambda x: 1 if x == "Vero" else 0)

        # Calcola la somma cumulativa della colonna 'Conta Vero' e crea la colonna 'Somma Cumulativa'
        df['Somma Cumulativa'] = df['Conta Vero'].cumsum()

        # Seleziona l'intervallo di regressione
        period_options = [38, 76, 114]
        selected_period = st.selectbox("Seleziona l'intervallo di regressione:", period_options)

        # Inizializza le colonne per Regressione e Intercetta
        df['Regressione'] = np.nan
        df['Intercetta'] = np.nan

        # Calcola i valori previsti per il periodo selezionato
        df = calculate_forecast(df, selected_period)

        # Calcola la differenza tra Somma Cumulativa e Valori Previsti
        df['Differenza'] = df['Somma Cumulativa'] - df['Valori Previsti']

        # Calcola le bande di deviazione standard
        bands = calculate_bollinger_bands(df['Differenza'].dropna())
        df['mean'] = bands['mean']
        df['upper_1std'] = bands['upper_1std']
        df['lower_1std'] = bands['lower_1std']
        df['upper_2std'] = bands['upper_2std']
        df['lower_2std'] = bands['lower_2std']
        df['upper_3std'] = bands['upper_3std']
        df['lower_3std'] = bands['lower_3std']

        # Visualizza la tabella aggiornata con le colonne 'Regressione', 'Intercetta', 'Valori Previsti', e 'Differenza'
        show_table = st.checkbox('Mostra Tabella', value=False)
        if show_table:
            st.write("Tabella con 'Regressione', 'Intercetta', 'Valori Previsti', e 'Differenza':")
            st.dataframe(df[['Prima Colonna', 'Somma Cumulativa', 'Regressione', 'Intercetta', 'Valori Previsti', 'Differenza', 'mean', 'upper_1std', 'lower_1std', 'upper_2std', 'lower_2std', 'upper_3std', 'lower_3std']])

        # Grafico della differenza e delle bande di deviazione standard
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Differenza'], mode='markers+lines', name='Differenza'))
        fig.add_trace(go.Scatter(x=df.index, y=df['mean'], mode='lines', name='Media'))
        fig.add_trace(go.Scatter(x=df.index, y=df['upper_1std'], mode='lines', name='Upper 1 Std Dev'))
        fig.add_trace(go.Scatter(x=df.index, y=df['lower_1std'], mode='lines', name='Lower 1 Std Dev'))
        fig.add_trace(go.Scatter(x=df.index, y=df['upper_2std'], mode='lines', name='Upper 2 Std Dev'))
        fig.add_trace(go.Scatter(x=df.index, y=df['lower_2std'], mode='lines', name='Lower 2 Std Dev'))
        fig.add_trace(go.Scatter(x=df.index, y=df['upper_3std'], mode='lines', name='Upper 3 Std Dev'))
        fig.add_trace(go.Scatter(x=df.index, y=df['lower_3std'], mode='lines', name='Lower 3 Std Dev'))
        fig.update_layout(title='Differenza e Bande di Deviazione Standard', xaxis_title='Indice', yaxis_title='Valore')
        st.plotly_chart(fig, use_container_width=True)

        # Calcola l'istogramma
        bin_size = 0.2
        hist, bin_edges = create_histogram(df['Differenza'].dropna(), bin_size)
        
        # Grafico dell'istogramma
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=bin_edges[:-1],
            y=hist,
            width=[bin_edges[i + 1] - bin_edges[i] for i in range(len(bin_edges) - 1)],
            name='Istogramma Differenza',
            text=[f'{bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}' for i in range(len(bin_edges) - 1)],
            texttemplate='%{text}',
            textposition='outside'
        ))
        fig_hist.update_layout(title='Istogramma della Differenza', xaxis_title='Intervallo', yaxis_title='Frequenza')

        # Calcola le percentuali
        percentages = calculate_percentage_smaller(df, bin_edges)

        # Grafico a barre per le percentuali
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=bin_edges[:-1],
            y=percentages,
            width=[bin_edges[i + 1] - bin_edges[i] for i in range(len(bin_edges) - 1)],
            name='Percentuali Maggiori del Successivo',
            text=[f'{bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}' for i in range(len(bin_edges) - 1)],
            texttemplate='%{text}',
            textposition='outside'
        ))
        fig_bar.update_layout(title='Percentuali di Valori Minori del Successivo', xaxis_title='Intervallo', yaxis_title='Percentuale (%)')
        
        # Calcola il conteggio degli intervalli di deviazione standard
        counts = count_terms_in_intervals(df)
        intervals = ['lower_3std_lower_2std', 'lower_2std_lower_1std', 'lower_1std_upper_1std', 'upper_1std_upper_2std', 'upper_2std_upper_3std']
        count_values = [counts.get(interval, 0) for interval in intervals]

        # Grafico a barre per i conteggi negli intervalli di deviazione standard
        fig_count = go.Figure()
        fig_count.add_trace(go.Bar(
            x=intervals,
            y=count_values,
            name='Conteggio Differenza negli Intervalli',
            text=[f'{count}' for count in count_values],
            texttemplate='%{text}',
            textposition='outside'
        ))
        fig_count.update_layout(title='Conteggio di Differenza negli Intervalli di Deviazione Standard', xaxis_title='Intervallo', yaxis_title='Conteggio')

        # Calcola il conteggio degli intervalli di deviazione standard
        counts = count_terms_in_intervals(df)
        intervals = ['lower_3std_lower_2std', 'lower_2std_lower_1std', 'lower_1std_upper_1std', 'upper_1std_upper_2std', 'upper_2std_upper_3std']
        count_values = [counts.get(interval, 0) for interval in intervals]

        # Grafico a barre per i conteggi negli intervalli di deviazione standard
        fig_count = go.Figure()
        fig_count.add_trace(go.Bar(
            x=intervals,
            y=count_values,
            name='Conteggio Differenza negli Intervalli',
            text=[f'{count}' for count in count_values],
            texttemplate='%{text}',
            textposition='outside'
        ))
        fig_count.update_layout(title='Conteggio di Differenza negli Intervalli di Deviazione Standard', xaxis_title='Intervallo', yaxis_title='Conteggio')

        # Calcola la percentuale rispetto al totale
        percentage_counts = calculate_percentage_of_total(counts)
        percentage_values = [percentage_counts.get(interval, 0) for interval in intervals]

# Intestazioni degli assi orizzontali
interval_labels = [
    "-3σ / -2σ",
    "-2σ / -1σ",
    "-1σ / +1σ",
    "+1σ / +2σ",
    "+2σ / +3σ"
]

# Grafico a barre per i conteggi negli intervalli di deviazione standard
fig_count.update_layout(
    title='Conteggio di Differenza negli Intervalli di Deviazione Standard',
    xaxis_title='Intervallo',
    xaxis=dict(tickvals=list(range(len(interval_labels))), ticktext=interval_labels),
    yaxis_title='Conteggio'
)

# Grafico a barre per le percentuali negli intervalli di deviazione standard
fig_percentage = go.Figure()
fig_percentage.add_trace(go.Bar(
    x=interval_labels,
    y=percentage_values,
    name='Percentuali di Differenza',
    text=[f'{percentage:.2f}%' for percentage in percentage_values],
    texttemplate='%{text}',
    textposition='outside'
))
fig_percentage.update_layout(
    title='Percentuali di Differenza negli Intervalli di Deviazione Standard',
    xaxis_title='Intervallo',
    xaxis=dict(tickvals=list(range(len(interval_labels))), ticktext=interval_labels),
    yaxis_title='Percentuale (%)'
)

# Ottieni l'ultimo valore di Differenza
last_difference = df['Differenza'].iloc[-1]

# Trova il centro dei bin
bin_edges_centered = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

# Trova il bin più vicino all'ultimo valore di 'Differenza'
closest_bin = min(bin_edges_centered, key=lambda x: abs(x - last_difference))

# Evidenzia il bin corretto colorandolo di rosso sia in fig.hist che in fig.bar
fig_hist, fig_bar = highlight_bin(fig_hist, fig_bar, closest_bin, bin_edges, hist_color='blue', highlight_color='red')

# Calcola le bande di deviazione standard
bands = calculate_bollinger_bands(df['Differenza'])

# Condizione 1: value < lower_2std
consecutive_counts_1 = count_consecutive_conditions(df['Differenza'], bands, condition1, opposite_condition1)

# Condizione 2: lower_2std <= value < lower_1std
consecutive_counts_2 = count_consecutive_conditions(df['Differenza'], bands, condition2, opposite_condition2)

# Condizione 3: upper_1std <= value < upper_2std
consecutive_counts_3 = count_consecutive_conditions(df['Differenza'], bands, condition3, opposite_condition3)

# Condizione 4: value >= upper_2std
consecutive_counts_4 = count_consecutive_conditions(df['Differenza'], bands, condition4, opposite_condition4)


fig1 = plot_consecutive_histogram(consecutive_counts_1, "Condizione 1", "Termini consecutivi per Condizione 1")
fig2 = plot_consecutive_histogram(consecutive_counts_2, "Condizione 2", "Termini consecutivi per Condizione 2")
fig3 = plot_consecutive_histogram(consecutive_counts_3, "Condizione 3", "Termini consecutivi per Condizione 3")
fig4 = plot_consecutive_histogram(consecutive_counts_4, "Condizione 4", "Termini consecutivi per Condizione 4")


# Visualizza i grafici affiancati e posiziona il grafico dei conteggi sotto l'istogramma della differenza
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_hist, use_container_width=True)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.plotly_chart(fig_count, use_container_width=True)
    st.plotly_chart(fig_percentage, use_container_width=True)

# Visualizzazione dei grafici due a due
figures = st.columns(2)

with figures[0]:
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

with figures[1]:
    st.plotly_chart(fig3)
    st.plotly_chart(fig4)
