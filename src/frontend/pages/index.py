import sys
from pathlib import Path
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

sys.path.append(str(Path().absolute()))
from src.backend.module.read_dataset import read_dataset


def initialize_state():
    st.session_state["show_prophet"] = 0
    st.session_state["show_table"] = 0


def show_table(input_file: UploadedFile) -> None:
    st.markdown("### 📈インプットデータの表示")
    if st.session_state["show_table"] > 0:
        st.dataframe(input_file)
    if st.sidebar.button(label="非表示", key="テーブル非表示"):
        st.session_state["show_status"] = 0


def render():
    initialize_state()
    st.title("工場入荷予測モデル")
    input_file = st.sidebar.file_uploader(label="インプットファイルをアップロードしてください。", type=["csv"])
    if input_file:
        file = read_dataset(input_file, "day")
        if st.sidebar.button(label="インプットデータの表示", key="インプットデータの表示"):
            st.session_state["show_table"] = 1
            show_table(file)


render()
