#Linear regression with numpy (without pytorch)
import numpy as np 

# lets y= 2*x
x= np.array([1,2,3,4], dtype=np.float32)
y= np.array([2,4,6,8], dtype=np.float32)
print(x, y)
w= 0.0

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

for epoch in range(n_iters):
    y_pred= forward(x)
    c= cost(y, y_pred)
    
    c.backward()
    with torch.no_grad():
        w-= learning_rate*w.grad
    w.grad.zero_()
    
    if epoch%5== 0:
        print(f"Epoch {epoch+1}: w= {w.item():.3f}, c= {c:.8f}")