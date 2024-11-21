import numpy as np 
import matplotlib.pyplot as plt 
 
def f(x): 
    return np.exp(x) + np.arctan(x) 
 
def solve_gauss(A, b): 
    n = len(A) 
    for i in range(n): 
        max_row = i + np.argmax(np.abs(A[i:, i])) 
        A[[i, max_row]] = A[[max_row, i]] 
        b[[i, max_row]] = b[[max_row, i]] 
        A[i] = A[i] / A[i, i] 
        b[i] = b[i] / A[i, i] 
         
        for j in range(i + 1, n): 
            b[j] -= b[i] * A[j, i] 
            A[j] -= A[i] * A[j, i] 
     
    x = np.zeros(n) 
    for i in range(n - 1, -1, -1): 
        x[i] = b[i] - np.sum(A[i, i + 1:] * x[i + 1:]) 
    return x 
 
def cubic_spline(x_nodes, y_nodes, degree=3, defect=0): 
    n = len(x_nodes) - 1 
    h = np.diff(x_nodes) 
     
    if degree > 3 or degree < 1:
        raise ValueError("Degree must be 1, 2, or 3")
    
    if defect < 0 or defect >= degree:
        raise ValueError(f"Defect must be between 0 and {degree-1}")
    
    if degree == 3:
        alpha = np.zeros(n - 1) 
        for i in range(1, n): 
            alpha[i - 1] = (3 / h[i] * (y_nodes[i + 1] - y_nodes[i]) - 
                            3 / h[i - 1] * (y_nodes[i] - y_nodes[i - 1])) 
         
        A = np.zeros((n - 1, n - 1)) 
        for i in range(n - 1): 
            if i > 0: 
                A[i, i - 1] = h[i] 
            A[i, i] = 2 * (h[i] + h[i + 1] if i < n - 2 else h[i]) 
            if i < n - 2: 
                A[i, i + 1] = h[i + 1] 
         
        c = np.zeros(n + 1) 
        if n > 1: 
            c[1:n] = solve_gauss(A, alpha) 
         
        b = np.zeros(n) 
        d = np.zeros(n) 
        for i in range(n): 
            b[i] = ((y_nodes[i + 1] - y_nodes[i]) / h[i] - 
                    h[i] * (c[i + 1] + 2 * c[i]) / 3) 
            d[i] = (c[i + 1] - c[i]) / (3 * h[i]) 
        
        return y_nodes[:-1], b, c[:-1], d, x_nodes
    
    elif degree == 2:
        b = (y_nodes[1:] - y_nodes[:-1]) / (x_nodes[1:] - x_nodes[:-1])
        c = np.zeros_like(b)
        d = np.zeros_like(b)
        return y_nodes[:-1], b, c, d, x_nodes
    
    elif degree == 1:
        b = np.zeros_like(y_nodes[:-1])
        c = np.zeros_like(b)
        d = np.zeros_like(b)
        return y_nodes[:-1], b, c, d, x_nodes 
 
def evaluate_spline(x, coeffs): 
    a, b, c, d, x_nodes = coeffs 
    for i in range(len(x_nodes) - 1): 
        if x_nodes[i] <= x <= x_nodes[i + 1]: 
            dx = x - x_nodes[i] 
            return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3 
    return None 
 
def spline_to_string(coeffs): 
    a, b, c, d, x_nodes = coeffs 
    polynomials = [] 
    for i in range(len(a)): 
        poly = f"S_{i}(x) = {a[i]:.6f}" 
        if b[i] != 0:
            poly += f" + {b[i]:.6f}*(x - {x_nodes[i]:.6f})" 
        if c[i] != 0:
            poly += f" + {c[i]:.6f}*(x - {x_nodes[i]:.6f})^2" 
        if d[i] != 0:
            poly += f" + {d[i]:.6f}*(x - {x_nodes[i]:.6f})^3" 
        poly += f", для x ∈ [{x_nodes[i]:.6f}, {x_nodes[i+1]:.6f}]" 
        polynomials.append(poly) 
    return polynomials 
 
a, b = -1, 1 
print("Введіть кількість вузлів n:") 
n = int(input()) 

print("Введіть степінь сплайна (1, 2 або 3):") 
degree = int(input()) 

print("Введіть дефект сплайна :") 
defect = int(input()) 
 
x_nodes = np.linspace(a, b, n) 
y_nodes = f(x_nodes) 
 
coeffs = cubic_spline(x_nodes, y_nodes, degree, defect) 
 
polynomials = spline_to_string(coeffs) 
print("\nПобудовані поліноми сплайна:") 
for poly in polynomials: 
    print(poly) 
 
x_values = np.linspace(a, b, 500) 
f_values = f(x_values) 
spline_values = [evaluate_spline(x, coeffs) for x in x_values] 
 
plt.figure(figsize=(10, 8)) 
 
plt.subplot(2, 1, 1) 
plt.plot(x_values, f_values, label='f(x)', color='blue') 
plt.plot(x_values, spline_values, label='Сплайн', linestyle='--', color='green') 
plt.scatter(x_nodes, y_nodes, color='red', zorder=5, label='Вузли') 
plt.title(f'f(x) та {degree}-ий сплайн') 
plt.xlabel('x') 
plt.ylabel('y') 
plt.grid(True) 
plt.legend() 
 
plt.subplot(2, 1, 2) 
error_values = f_values - np.array(spline_values) 
plt.plot(x_values, error_values, label=f'f(x) - {degree}-ий Сплайн', color='purple') 
plt.title('Похибка сплайна') 
plt.xlabel('x') 
plt.ylabel('f(x) - Сплайн') 
plt.grid(True) 
plt.legend() 
 
plt.tight_layout() 
plt.show() 
 
print(f"\nМаксимальна похибка: {np.max(np.abs(error_values)):.6f}")