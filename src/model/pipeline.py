import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression


def reshape_visual_features(x):
    return np.array(x.tolist()).reshape(-1, 4)


def build_model_pipeline():
    text_pipe = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                strip_accents="unicode",
                lowercase=True,
            ),
        ),
    ])

    visual_pipe = Pipeline([
        ("reshape", FunctionTransformer(reshape_visual_features, validate=False)),
        ("scaler", RobustScaler()),
    ])

    model = Pipeline([
        (
            "features",
            ColumnTransformer(
                [
                    ("text", text_pipe, "Processed_Text"),
                    ("visual", visual_pipe, "Visual_Features"),
                ],
                remainder="drop",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="saga",
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ])

    return model