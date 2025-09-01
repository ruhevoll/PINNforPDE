import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class ODE_nth(tf.keras.Model):
    def __init__(self, ode_function, conditions, n, ode_type='IVP'):
        super(ODE_nth, self).__init__()
        self.ode_function = ode_function
        self.conditions = conditions # List of boundary poin
        self.n = n # Order of the ODE
        self.ode_type = ode_type #IVP/BVP
        # Define neural network architecture
        self.dense1 = tf.keras.layers.Dense(50, activation='elu')
        self.dense2 = tf.keras.layers.Dense(50, activation='elu')
        self.dense3 = tf.keras.layers.Dense(50, activation='elu')
        self.dense4 = tf.keras.layers.Dense(1)
        self.mse_loss = tf.keras.losses.MeanSquaredError() # Define MSE loss

    def call(self, inputs, training=False):
        x = tf.reshape(inputs, [-1, 1]) # Ensure input has shape (batch_size, 1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x

    def train_step(self, data):
        x, y_exact = data
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y_exact = tf.convert_to_tensor(y_exact, dtype=tf.float32)
        # Compute derivatives
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x) # "Tracks" computations involving x for derivative calculations
            y_NN = self(x, training=True) # Computes NN output y_NN(x) at training points x, set training = True
            derivatives = [y_NN] # Stores derivatives [y_NN, y_NN', y_NN'', ... y_NN(n)]
            for _ in range(self.n):
                y_NN = derivatives[-1] # Takes last computed derivative as the input for the next
                dy_dx = tape.gradient(y_NN, x) # Gradient computer
                if dy_dx is None:
                    raise ValueError("Gradient dy_dx is None; check tensor tracking.")
                derivatives.append(dy_dx) # Adds gradient to the list
            # ODE loss function!!
            ode_residual = self.ode_function(x, *derivatives)
            loss = self.mse_loss(ode_residual, tf.zeros_like(ode_residual)) # This loss function measures the MSE between the residual and the tensor of zeros. I.e., F(x, y_NN, ... y''..(n)..''_NN) = 0
            # IVP/BVP dependent loss function
            for x_c, y_c, order in self.conditions:
                x_c_tf = tf.convert_to_tensor([x_c], dtype=tf.float32)
                tape.watch(x_c_tf)
                y_c_NN = self(x_c_tf, training=True)
                for _ in range(order):
                    y_c_NN = tape.gradient(y_c_NN, x_c_tf)
                    if y_c_NN is None:
                        raise ValueError("Boundary condition gradient is None; check tensor tracking.")
                loss += self.mse_loss(y_c_NN, tf.convert_to_tensor([y_c], dtype=tf.float32))
        # Update weights
        gradients = tape.gradient(loss, self.trainable_weights)
        if gradients is None or any(g is None for g in gradients):
            raise ValueError("Gradients are None; check computational graph.")
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Update metrics
        self.compiled_metrics.update_state(y_exact, self(x, training=True))
        return {m.name: m.result() for m in self.metrics}

# Define IVP F(x, y, y', y'', ...)
def ode_function_ivp(x, y, dy_dx, d2y_dx2):
    return y*dy_dx - 4*x

ivp_conditions = [(1.0, 5, 0), (1.0,2.0,1)] # Format: List of tuples (x_c, y_c, order) where x_c is the point, y_c is the value, and order is the derivative order
model_ivp = ODE_nth(ode_function_ivp, ivp_conditions, n=2, ode_type='IVP')

# Compile and train IVP
x_train = np.linspace(0, 4, 200).reshape(-1, 1)
y_train = (x_train**2 + 4).reshape(-1, 1) # Exact solution
model_ivp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0005), loss='mse', metrics=['mse'])
history_ivp = model_ivp.fit(x_train, y_train, batch_size=1, epochs=40)

"""
# Example usage for BVP: y'' + y = 0, y(0) = 1, y(π/2) = 0
def ode_function_bvp(x, y, dy_dx, d2y_dx2):
    return d2y_dx2 + y

bvp_conditions = [(0.0, 1.0, 0), (np.pi/2, 0.0, 0)] # y(0) = 1, y(π/2) = 0
model_bvp = ODE_nth(ode_function_bvp, bvp_conditions, n=2, ode_type='BVP')

# Compile and train BVP
x_train_bvp = np.linspace(0, np.pi/2, 20).reshape(-1, 1)
y_train_bvp = np.cos(x_train_bvp).reshape(-1, 1) # Exact solution
model_bvp.compile(optimizer='adam', loss='mse', metrics=['mse'])
history_bvp = model_bvp.fit(x_train_bvp, y_train_bvp, batch_size=1, epochs=40)

# Summarize history for loss and metrics (IVP)
plt.rcParams['figure.dpi'] = 150
plt.plot(history_ivp.history['loss'], color='magenta', label='Total losses ($L_D + L_B$)')
plt.plot(history_ivp.history['mse'], color='cyan', label='MSE')
plt.yscale("log")
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.title('IVP Training History')
plt.show()
"""

# Check the PINN at different points not included in the training set
n = 500
x = np.linspace(0, 4, n) # Adjust for domain, e.g., for IVP
y_exact = (x**2 + 4) # Exact solution
x_tf = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float32)
with tf.GradientTape(persistent=True) as t:
    t.watch(x_tf)
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(x_tf)
        y_NN = model_ivp(x_tf) # Use model_ivp or model_bvp
    dy_dx_NN = t2.gradient(y_NN, x_tf)
d2y_dx2_NN = t.gradient(dy_dx_NN, x_tf)
# Plot the results
plt.rcParams['figure.dpi'] = 150
plt.plot(x, y_exact, color="black", linestyle='solid', linewidth=2.5, label="$y(x)$ analytical")
plt.plot(x, y_NN, color="red", linestyle='dashed', linewidth=2.5, label="$y_{NN}(x)$")
plt.plot(x, dy_dx_NN, color="blue", linestyle='-.', linewidth=3.0, label="$y'_{NN}(x)$")
plt.plot(x, d2y_dx2_NN, color="green", linestyle='dotted', linewidth=3.0, label="$y''_{NN}(x)$")
plt.legend()
plt.xlabel("x")
plt.title('IVP Solution')
plt.show()

"""
# Summarize history for loss and metrics (BVP)
plt.rcParams['figure.dpi'] = 150
plt.plot(history_bvp.history['loss'], color='magenta', label='Total losses ($L_D + L_B$)')
plt.plot(history_bvp.history['mse'], color='cyan', label='MSE')
plt.yscale("log")
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.title('BVP Training History')
plt.show()

# For BVP plotting (adjust domain)
n = 500
x = np.linspace(0, np.pi/2, n) # Adjust for domain, e.g., for BVP
y_exact = np.cos(x) # Exact solution for y'' + y = 0
x_tf = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float32)
with tf.GradientTape(persistent=True) as t:
    t.watch(x_tf)
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(x_tf)
        y_NN = model_bvp(x_tf) # Use model_ivp or model_bvp
    dy_dx_NN = t2.gradient(y_NN, x_tf)
d2y_dx2_NN = t.gradient(dy_dx_NN, x_tf)
# Plot the results
plt.rcParams['figure.dpi'] = 150
plt.plot(x, y_exact, color="black", linestyle='solid', linewidth=2.5, label="$y(x)$ analytical")
plt.plot(x, y_NN, color="red", linestyle='dashed', linewidth=2.5, label="$y_{NN}(x)$")
plt.plot(x, dy_dx_NN, color="blue", linestyle='-.', linewidth=3.0, label="$y'_{NN}(x)$")
plt.plot(x, d2y_dx2_NN, color="green", linestyle='dotted', linewidth=3.0, label="$y''_{NN}(x)$")
plt.legend()
plt.xlabel("x")
plt.title('BVP Solution')
plt.show()
"""