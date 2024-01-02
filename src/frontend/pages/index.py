import sys
from pathlib import Path
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

sys.path.append(str(Path().absolute()))
from src.backend.module.read_dataset import read_dataset
from src.backend import main as backend_main

from src.frontend.component.show_explanations import (
    show_table,
    show_prophet,
    show_shap_importance,
)


def initialize_state():
    st.session_state["show_table"] = 0
    st.session_state["show_prophet"] = 0
    st.session_state["show_shap_importance"] = 0


def regressor(input_file: UploadedFile) -> None:
    df, holidays_df = backend_main.handle_dataset(input_file, "day")
    prophet_model, df_with_prophet, pred = backend_main.prophet(
        df, holidays_df, "Yシャツ", "day"
    )
    return backend_main.pred_random_forest(df_with_prophet, "Yシャツ")


def render():
    initialize_state()
    st.title("工場入荷予測モデル")
    input_file = st.sidebar.file_uploader(label="インプットファイルをアップロードしてください。", type=["csv"])
    if input_file:
        if st.sidebar.button(label="インプットデータの表示", key="インプットデータの表示"):
            st.session_state["show_table"] = 1
            show_table(input_file)
        if st.sidebar.button(label="時系列要素の表示", key="時系列要素の表示"):
            st.session_state["show_prophet"] = 1
            show_prophet(input_file)
        if st.sidebar.button(label="特徴量貢献度の表示", key="特徴量貢献度の表示"):
            st.session_state["show_shap_importance"] = 1
            rf_model, x_train, r2_score = regressor(input_file)
            show_shap_importance(rf_model, x_train)


render()
