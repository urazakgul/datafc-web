import streamlit as st
import pandas as pd
from typing import Tuple, Optional, List, Union

def require_session_data(*keys: str) -> Tuple[Optional[pd.DataFrame], ...]:
    missing = []
    dfs = []

    for key in keys:
        df = st.session_state.get(key)
        if df is None or not hasattr(df, "empty") or df.empty:
            missing.append(key.replace("_", " "))
            dfs.append(None)
        else:
            dfs.append(df)

    if missing:
        st.warning(f"No data found for: {', '.join(missing)}")
        st.stop()

    return tuple(dfs)

def filter_matches_by_status(df: pd.DataFrame, statuses: Union[str, List[str]]) -> pd.DataFrame:
    if "status" not in df.columns:
        return df

    if isinstance(statuses, str):
        statuses = [statuses]

    return df[df["status"].isin(statuses)]