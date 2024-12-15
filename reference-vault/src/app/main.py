from capstone.mpg_regression_library import main as mpglib
from capstone.mpg_regression_manual import main as mpgmanual
from capstone.tumor_classification_library import main as tumlib
from capstone.tumor_classification_manual import main as tummanual
from capstone.handwrite import main as handwrite


def main() -> None:
    print("======== regresion lineal libreria MPG example ========")
    mpglib()
    print("======== regresion lineal manual MPG example ========")
    mpgmanual()
    print("======== regresion logistica libreria Tumor example ========")
    tumlib()
    print("======== regresion logistica manual Tumor example ========")
    tummanual()
    print("======== redes neuronales pytorch handwrite example ========")
    handwrite()


if __name__ == "__main__":
    main()
