import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        pass

    def run(self):
        pass


def plot(data):
    plt.ylabel('Age')
    plt.xlabel('Charges')
    plt.plot(np.arange(len(data)), data)
    plt.legend()


if __name__ == '__main__':
    ages = np.array([18, 58, 23, 45, 63, 36])
    BMIs = np.array([53.13, 49.06, 17.38, 21, 21.66, 28.59])
    charges = np.array([1163.43, 11381.33, 2775, 7222, 14349, 6548])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('Age')
    ax.set_ylabel('BMI')
    ax.set_zlabel('Charges')
    ax.scatter(ages, BMIs, charges, marker='o')
    # plt.legend()
    plt.show()
