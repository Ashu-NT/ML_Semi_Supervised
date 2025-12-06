import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline

def reshape_visual_features(x):
    return np.array(x.tolist()).reshape(-1, 4)

def build_model_pipeline():
    text_pipe = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=25000,
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9,
                sublinear_tf=True,
            ),
        ),
        ("svd", TruncatedSVD(n_components=100, random_state=42)),
    ])

    visual_pipe = Pipeline([
        ("reshape", FunctionTransformer(reshape_visual_features)),
        ("scaler", RobustScaler()),
    ])

    model = make_imb_pipeline(
        ColumnTransformer(
            [
                ("text", text_pipe, "Processed_Text"),
                ("visual", visual_pipe, "Visual_Features"),
            ]
        ),
        SMOTE(random_state=42),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ),
    )
    return model
