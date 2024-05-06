import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

class SimpleLinearRegression:
    """
    Oddiy logistik regressiya
    """
    def __init__(self):
        self.w = 0
        self.b = 0
    
    def predict(self, x):
        """Bashorat qiliuvchi funksiya

        Args:
            x (_type_): erkli o'zgaruvchi

        Returns:
            _type_: erksiz o'zgaruvchining qiymatini qaytaradi
        """
        if isinstance(x, list):
            return [self.w * float(xi) + self.b for xi in x]
        else:
            return self.w * float(x) + self.b
    
    def loss(self, x, y):
        """Xatolikni hisoblochi funksiya

        Args:
            x (_type_): erksiz o'zgaruvchi 
            y (_type_): erkli o'zgaruchi

        Returns:
            _type_: _description_
        """
        total_error = sum((y[i] - self.predict(x[i]))**2 for i in range(len(x)))
        return total_error / len(x)
    
    def update_weights(self, x, y, lr):
        """w va b ni yangilovchi funksiya

        Args:
            x (_type_): erkli o'zgaruchi
            y (_type_): erksiz o'zgaruvchi
            lr (_type_): o'qitish qadami
        """
        w_deriv = sum(-2 * float(x[i]) * (y[i] - self.predict(x[i])) for i in range(len(x)))
        b_deriv = sum(-2 * (y[i] - self.predict(x[i])) for i in range(len(x)))
        self.w -= (w_deriv / float(len(x))) * lr
        self.b -= (b_deriv / float(len(x))) * lr

    def train(self, x, y, lr=0.8,n=1000,epsilon=1000):
        """Modelni train qilivuvchi funksiya

        Args:
            x (_type_): erkli o'zgaruvchi
            y (_type_): erksiz o'zgaruvchi
            lr (_type_): o'qitish qadami
            n (_type_): iteratsiyalar soni

        Returns:
            _type_: w, loss, epoch
        """
        losses = []
        weights = []
        epoch = 1
        
        while epoch < n:
            self.update_weights(x, y, lr)
            current_loss = self.loss(x, y)
            losses.append(current_loss)
            weights.append(self.w)
            epoch += 1
            
            if current_loss < epsilon:
                epoch += 1
                return weights, losses, epoch
            
            
            
        return weights, losses, epoch
    
linear = SimpleLinearRegression()

# Faydan datasetni o'qib oliyapmiz
x_data = pd.read_csv("x_data.csv").iloc[:, 0].tolist()
y_data = pd.read_csv("y_data.csv").iloc[:, 0].tolist()

# print("x_data:",x_data)
# print("y_data:",y_data)

epsilon_value = float(input("Epsilon : "))
lr = float(input("Learning rate:"))
n = int(input("N="))
weights, losses, epochs = linear.train(x_data, y_data, lr=lr,n=n,epsilon=epsilon_value)

#Plot data points and the linear regression line
plt.scatter(x_data, y_data, label="Ma'lumot nuqtalari")
plt.plot(x_data, linear.predict(x_data), color='red', label='Linear Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Data Points and Linear Regression Line')
plt.show()

# Print final weight, bias, number of epochs, and a prediction example
print("Function: y =", str(linear.w) + " x + " + str(linear.b))
print("Weight:", linear.w)
print("Bias:", linear.b)
print("Number of epochs:", epochs)
print("Bashorat: (x=1.5)",linear.predict(x=1.5))


# X va y datasetni beryapmiz
X = np.array(x_data).reshape(-1,1)
y = np.array(y_data)

model = LinearRegression()

# Modelni o'qitish
model.fit(X, y)

# Modelni bashoratlash
predictions = model.predict(X)
print("Bashorat:",model.predict(X=np.array(1.5).reshape(-1,1)))


# Modelni chizish
plt.scatter(X, y, color='green', label='Haqiqiy nuqtalar')
plt.plot(X, predictions, color='red', label='Library Linear Regression')
plt.plot(x_data, linear.predict(x_data), color='blue', label='My Linear Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
