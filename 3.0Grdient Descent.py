#Linear regression with numpy (without pytorch)
import numpy as np 

# lets y= 2*x
x= np.array([1,2,3,4], dtype=np.float32)
y= np.array([2,4,6,8], dtype=np.float32)
print(x, y)
w= 0.0
c_h=[]
w_h=[]

# forward
def forward(x):
    return w*x

# cost
def cost(y, y_pred):
    return ((y-y_pred)**2).mean()

# gradient cal (dJ/dw = 1/N * -2x(y-w*x)
def gradient(y, y_pred, x):
    return np.mean(-(2*x*(y-y_pred))) #-(2*x*(y-y_pred))).mean() same

print(f"Before training: f(5)= {forward(5):.3f}")

# training
learning_rate= 0.01
n_iters= 61

for epoch in range(n_iters):
    # forward
    y_pred= forward(x)

    # cost
    c= cost(y, y_pred)
    c_h.append(c)
    w_h.append(w)


    #gradient cal
    dw= gradient(y, y_pred, x)

    # upadate
    w-= learning_rate*dw

    if epoch%5== 0:
        print(f"Epoch {epoch+1}: w= {w:.3f}, cost= {c:.8f}")

print(f"After trainig: f(5)= {forward(5):.3f}\n")


#Linear regression with pytorch
import torch

x= torch.tensor([1,2,3,4], dtype= torch.float)
y= torch.tensor([2,4,6,8], dtype= torch.float)
w= torch.tensor([0], dtype= torch.float, requires_grad= True)

print(x,y,w)

def forward(x):
    return w*x

def cost(y, y_pred):
    return ((y-y_pred)**2).mean() #torch.mean((y-y_pred)**2)

print(f"Before trainig: f(5): {forward(5).item():.3f}")

learning_rate= 0.01
n_iters= 61
cost
for epoch in range(n_iters):
    y_pred= forward(x)
    c= cost(y, y_pred)

    c.backward()
    with torch.no_grad():
        w-= learning_rate*w.grad
    w.grad.zero_()
    
    if epoch%5== 0:
        print(f"Epoch {epoch+1}: w= {w.item():.3f}, c= {c:.8f}")

print(f"After training: f(5)= {forward(5).item():.3f}")

## 2D
import matplotlib.pyplot as plt


# Plot the function
plt.plot(w_h, c_h)
plt.title('Graph of my_function')
plt.xlabel('w')
plt.ylabel('J')
plt.grid(True)
plt.show()

plt.plot(np.arange(n_iters), c_h)
plt.title('Graph of my_function')
plt.xlabel('Iter')
plt.ylabel('J')
plt.grid(True)

## 3D
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define the function you want to plot
def my_function(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

# Define a function to create the 3D plot
def plot_3d_function(f, x_start, x_end, y_start, y_end, num_points):
    # Create an array of x and y values
    x = np.linspace(x_start, x_end, num_points)
    y = np.linspace(y_start, y_end, num_points)
    x, y = np.meshgrid(x, y)
    
    # Compute the corresponding z values using the function
    z = f(x, y)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')

    # Set plot labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Graph of the function')

    # Show the plot
    plt.show()

# Use the plot_3d_function with abstract parameters
plot_3d_function(my_function, -5, 5, -5, 5, 100)

