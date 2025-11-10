"""Main application module for the project."""


import pathlib as path

from EncoderLayer.Sdr import SDR






ROOT_PATH = path.Path(__file__).parent.parent.parent.parent

DATA_PATH = ROOT_PATH / "Data"


def main() -> None:
    """Main function to demonstrate InputHandler usage."""
    # Create an SDR instance demoing the encoder layer
    sdr_one = SDR([3,3])
    sdr_two = SDR([3,3])
    sdr_three = SDR([3,3])
    sdr_cat = SDR([9,3])

    sdr_one.set_dense([1,0,1,0,1,0,1,0,1])
    sdr_two.set_dense([0,1,0,1,0,1,0,1,0])
    sdr_three.set_dense([1,1,0,0,1,1,0,0,1])

    print("SDR One:")
    print(sdr_one)
    print("SDR Two:")
    print(sdr_two)
    print("SDR Three:")
    print(sdr_three)

    sdr_cat.concatenate([sdr_two,sdr_one, sdr_three], axis=0)
    print("Union of SDR One,SDR Two, and SDR Three:")
    print(sdr_cat)
    

    sdr_sparse = SDR([100,21])
    sdr_sparse.set_sparse([0,2,4,6,8,2000])


    print("Sparse SDR:")
    print(sdr_sparse)





    

    






if __name__ == "__main__":

    main()
