from kfp import dsl
from kfp import compiler


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["boto3", "pandas", "scikit-learn", "mlflow", "joblib"]
)
def validate_op():

    import os
    import boto3

    REPO = "https://github.com/piyushsms91/mlops-core-project.git"

    os.system(f"git clone {REPO}")

    s3 = boto3.client("s3")

    os.makedirs(
        "mlops-core-project/data/raw",
        exist_ok=True
    )

    s3.download_file(
        "piyushsms91-edu-mlproject1-342480083282",
        "reviews.csv",
        "mlops-core-project/data/raw/reviews.csv"
    )

    os.chdir("mlops-core-project")

    os.system("python src/data_validation.py")


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["boto3", "pandas", "scikit-learn"]
)
def preprocess_op():

    import os
    import boto3

    REPO = "https://github.com/piyushsms91/mlops-core-project.git"

    os.system(f"git clone {REPO}")

    s3 = boto3.client("s3")

    os.makedirs(
        "mlops-core-project/data/raw",
        exist_ok=True
    )

    s3.download_file(
        "piyushsms91-edu-mlproject1-342480083282",
        "reviews.csv",
        "mlops-core-project/data/raw/reviews.csv"
    )

    os.chdir("mlops-core-project")

    os.system("python src/preprocessing.py")


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["boto3", "pandas", "scikit-learn", "mlflow", "joblib"]
)
def train_op():

    import os
    import boto3

    REPO = "https://github.com/piyushsms91/mlops-core-project.git"

    os.system(f"git clone {REPO}")

    s3 = boto3.client("s3")

    os.makedirs(
        "mlops-core-project/data/raw",
        exist_ok=True
    )

    s3.download_file(
        "piyushsms91-edu-mlproject1-342480083282",
        "reviews.csv",
        "mlops-core-project/data/raw/reviews.csv"
    )

    os.chdir("mlops-core-project")

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.system("python src/preprocessing.py")

    os.system("python src/train.py")


@dsl.pipeline(
    name="loksai-app"
)
def pipeline():

    validate = validate_op()

    preprocess = preprocess_op().after(validate)

    train_op().after(preprocess)


if __name__ == "__main__":

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.yaml"
    )
