#Linear regression with numpy (without pytorch)
import numpy as np 
import matplotlib.pyplot as plt

# lets y= 2*x
x= np.array([1,2,3,4], dtype=np.float32)
y= np.array([3,5,9,12], dtype=np.float32)
print(x, y)
w1= 0.1
w2= 0.1
w3= [0.1, 0.1]
c_h= []
w_h1= []

# forward
def forward(x):
    return (w3[0]*(w1*x) + w3[1]*(w2*x))

# cost
def cost(y, y_pred):
    return ((y-y_pred)**2).mean()

# gradient cal (dJ/dw = 1/N * -2x(y-w*x)
def gradient(y, y_pred, x):
    return [np.mean((2*x*w1*(y_pred-y))), np.mean((2*x*w2*(y_pred-y))), np.mean((2*x*w3[1]*(y_pred-y))), np.mean((2*x*w3[0]*(y_pred-y)))]

print(f"Before training: f(5)= {forward(5):.3f}")

# training
learning_rate= 0.01
n_iters= 20

for epoch in range(n_iters):
    # forward
    y_pred= forward(x)

    # cost
    c= cost(y, y_pred)
    c_h.append(c)
    w_h1.append(w1)


    #gradient cal
    dw= gradient(y, y_pred, x)
    # print(dw)

    # upadate
    w3[0]-= learning_rate*dw[0]
    w3[1]-= learning_rate*dw[1]
    w2-= learning_rate*dw[2]
    w1-= learning_rate*dw[3]

    if epoch%2== 0:
        print(f"Epoch {epoch}: w1= {w1:.3f}, w2= {w2:.3f}, w3[0]= {w3[0]:.3f}, w3[1]= {w3[1]:.3f}, cost= {c:.8f}")

print(f"After trainig: f(5)= {forward(5):.3f}\n")

# # Plot the function 2D
# plt.plot(w_h1, c_h)
# plt.title('Graph of my_function')
# plt.xlabel('w')
# plt.ylabel('J')
# plt.grid(True)
# # plt.show()

# Plot 3D
import sympy as sp

def def_fun(x,y):
    a= 0
    w1, w2, w3a, w3b= sp.symbols('w1 w2 w3a w3b')
    j= 0
    for i in range(len(y)):
        expr= ((w3a*(w1*x[i]) + w3b*(w2*x[i])) - y[i])**2
        expanded_expr = sp.expand(expr)
        # Simplify the expression
        simplified_expr = sp.simplify(expanded_expr)
        j+= simplified_expr
        a+= 1
    
    return j/a

def my_function(w1, w2, w3):
    # z= []
    
    # for i in range(len(w1)):
    #     z.append((1/4)*(30*(w1[i]**2)*(w2[i]**2) - 120*w1[i]*w2[i] + 120))

    # return z
    return 0 #7.5*(w1**2)*w3a**2 + 15.0*w1*w2*w3a*w3b - 44.0*w1*w3a + 7.5*w2**2*w3b**2 - 44.0*w2*w3b + 64.75

def plot_3d_function(f, x_start, x_end, y_start, y_end, num_points):
    # Create an array of x and y values
    w1 = np.linspace(x_start, x_end, num_points)
    w2 = np.linspace(y_start, y_end, num_points)
    w1, w2= np.meshgrid(w1, w2)
    j= def_fun(x,y)
    print(j)
    # print(w1.shape)
    # Compute the corresponding z values using the function
    J = f(w1, w2)

    # Find the indices of the minimum value(s) in z
    min_indices = np.unravel_index(np.argmin(J), J.shape)
    min_x = w1[min_indices]
    min_y = w2[min_indices]
    min_z = J[min_indices]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(w1, w2, J, cmap='viridis')

    ax.scatter(min_x, min_y, min_z, color='red', s=100, label=f'Minimum Point: ({min_x:.2f}, {min_y:.2f}, {min_z:.2f})')


    # Set plot labels and title
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('J')
    ax.set_title('3D Graph of the function')


    ax.legend()


    # Show the plot
    plt.show()

# Use the plot_3d_function with abstract parameters
j= def_fun(x,y)
print(j)
# plot_3d_function(my_function, -5, 5, -5, 5, 100)
