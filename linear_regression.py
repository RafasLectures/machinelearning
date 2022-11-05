import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        self.w = np.array([])

    def train(self, X, y):
        x_trans = np.transpose(X)
        inverse = np.linalg.inv(np.matmul(x_trans, X))
        self.w = np.matmul(np.matmul(inverse, x_trans), y)

    def run_model(self, x):
        return np.matmul(x, self.w)



def plot(data):
    plt.ylabel('Age')
    plt.xlabel('Charges')
    plt.plot(np.arange(len(data)), data)
    plt.legend()


if __name__ == '__main__':
    ages = np.array([[18, 58, 23, 45, 63, 36]]).T
    BMIs = np.array([[53.13, 49.06, 17.38, 21, 21.66, 28.59]]).T
    charges = np.array([[1163.43, 11381.33, 2775, 7222, 14349, 6548]]).T

    data = np.append(ages, BMIs, axis=1)
    linear_regression = LinearRegression()
    linear_regression.train(data, charges)

    newInput = np.array([40, 32.5])
    newOutput = linear_regression.run_model(newInput)
    print(newOutput)


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('Age')
    ax.set_ylabel('BMI')
    ax.set_zlabel('Charges')
    ax.scatter(ages, BMIs, charges, marker='o')
    ax.scatter(newInput[0],newInput[1], newOutput, marker='x')
    # plt.legend()
    plt.show()
