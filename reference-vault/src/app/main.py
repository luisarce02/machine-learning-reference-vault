from capstone.mpg_regression_library import main as mpglib
from capstone.tumor_classification_library import main as tumlib


def main() -> None:
    file_path = "data/auto_mpg.csv"
    mpglib(file_path)
    file_path2 = "data/breast_cancer.csv"
    tumlib(file_path2)


if __name__ == "__main__":
    main()
