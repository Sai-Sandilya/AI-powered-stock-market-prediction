import pandas as pd

def screen_by_pe(df: pd.DataFrame, max_pe: float) -> pd.DataFrame:
    return df[df['pe_ratio'] <= max_pe]
