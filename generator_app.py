import streamlit as st
from generator import simulate
from generator import warmup



st.set_page_config(page_title="Generator danych syntetycznych", layout="centered") #ustawia tytuł i układ strony

st.title("Generator danych syntetycznych", text_alignment="center")

with st.container(border=True):
    st.header("Przeprowadź symulacje",text_alignment="center")
    n_iteration = st.number_input("Ilość rekordów", min_value=1, value=100, step=100)
    test_type = st.selectbox("Rodzaj", [0, 1], format_func=lambda x: "Punktacja standardowa" if x == 0 else "Punktacja ujemna")


warmup()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("START", use_container_width=True, disabled = st.session_state.get('run_sim', False)):
        st.session_state.run_sim = True
        st.rerun()
if st.session_state.get('run_sim'):
    st.session_state.run_sim = False 
    with st.spinner("Generowanie danych..."):
        df_final, duration = simulate(n_iteration, test_type)
    st.markdown(f"Generowanie zakończone! :tada:")
    st.markdown(f"**Czas trwania: {duration:.2f} sekund**")
    st.balloons()
    csv = df_final.to_csv(index=False).encode('utf-8')
    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            label="Pobierz plik CSV",
            data=csv,
            mime="text/csv",
            use_container_width=True,
            file_name=("symulacje_standardowe" if test_type == 0 else "symulacje_ujemne") + ".csv"
            ) 
    with col2:
        if st.button("Wygeneruj ponownie", use_container_width=True):
            st.rerun()

    