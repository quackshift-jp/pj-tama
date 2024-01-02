import pandas as pd
import sys
from pathlib import Path

import streamlit as st
import shap
from streamlit.runtime.uploaded_file_manager import UploadedFile
from sklearn.ensemble import RandomForestRegressor

sys.path.append(str(Path().absolute()))
from src.backend.module.read_dataset import read_dataset
from src.backend import main as backend_main


def show_table(input_file: UploadedFile) -> None:
    st.markdown("### 📄インプットデータの表示")
    file = read_dataset(input_file, "day")
    if st.session_state["show_table"] > 0:
        st.dataframe(file)
    if st.sidebar.button(label="非表示", key="テーブル非表示"):
        st.session_state["show_status"] = 0


def show_prophet(input_file: UploadedFile) -> None:
    st.markdown("### 📊時系列要素の表示")
    df, holidays_df = backend_main.handle_dataset(input_file, "day")
    prophet_model, df_with_prophet, pred = backend_main.prophet(
        df, holidays_df, "Yシャツ", "day"
    )
    if st.session_state["show_prophet"] > 0:
        st.pyplot(prophet_model.plot_components(pred))
    if st.sidebar.button(label="時系列要素の非表示", key="時系列要素の非表示"):
        st.session_state["show_prophet"] = 0


def show_shap_importance(rf_model: RandomForestRegressor, x_train: pd.DataFrame):
    st.markdown("### ✅特徴量貢献度の表示")
    if st.session_state["show_shap_importance"] > 0:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X=x_train)
        st.pyplot(shap.summary_plot(shap_values, x_train))
    if st.sidebar.button(label="特徴量貢献度の非表示", key="特徴量貢献度の非表示"):
        st.session_state["show_shap_importance"] = 0
