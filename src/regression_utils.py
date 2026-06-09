import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)

from sklearn.linear_model import LinearRegression


def load_data():

    return sns.load_dataset("diamonds")


def preprocess_data(df):

    X = df.drop("price", axis=1)

    y = df["price"]

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_cols
            )
        ],
        remainder="passthrough"
    )

    return X, y, preprocessor


def train_model():

    df = load_data()

    X, y, preprocessor = preprocess_data(df)

    base_models = [

        (
            "rf",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        ),

        (
            "gbr",
            GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]

    stacking_model = StackingRegressor(

        estimators=base_models,

        final_estimator=LinearRegression()
    )

    model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "regressor",
            stacking_model
        )
    ])

    model.fit(X, y)

    return model