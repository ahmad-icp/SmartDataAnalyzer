from ai_insights import generate_insights as suggest_ai_actions
from ai_insights import _local_insights
from cleaning_tools import fill_missing, convert_types, standardize_text_columns

def apply_ai_actions(df, action):
    # minimal compatibility wrapper used by app
    t = action.get("type")
    if t == "fill":
        strat = action.get("strategy", "median")
        return fill_missing(df, strategy=strat)
    if t == "convert":
        return convert_types(df, {action.get("col"): action.get("to")})
    if t == "standardize":
        return standardize_text_columns(df, [action.get("col")])
    return df

__all__ = ["suggest_ai_actions", "apply_ai_actions"]
