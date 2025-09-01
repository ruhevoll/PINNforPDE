import tensorflow as tf
import numpy as np

class ODE_nth(tf.keras.Model):
    def __init__(self, ode_function, conditions, n, ode_type = 'IVP'):
       super(ODE_nth, self).__init__()
       self.ode_function = ode_function
       self.conditions = conditions # List of boundary poin
       self.n = n # Order of the ODE
       self.ode_type = ode_type #IVP/BVP

    def train_step(self, data):
        x, y_exact = data

        # Compute derivatives
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(x)                                   # "Tracks" computations involving x for derivative calculations
            y_NN = self(x, training = True)                 # Computes NN output y_NN(x) at training points x, set training = True
            derivatives = [y_NN]                            # Stores derivatives [y_NN, y_NN', y_NN'', ... y_NN(n)]
            for _ in range (self.n):
                y_NN = derivatives[-1]                      # Takes last computed derivative as the input for the next
                dy_dx = tape.gradient(y_NN, x)              # Gradient computer
                derivatives.append(dy_dx)                   # Adds gradient to the list 

        # ODE loss function!!
        ode_residual = self.ode_function(x, *derivatives)
        loss = self.compiled_loss(ode_residual, tf.zeros_like(ode_residual)) # This loss function measures the MSE between the residual and the tensor of zeros. I.e., F(x, y_NN, ... y''..(n)..''_NN) = 0 

        # IVP/BVP dependent loss function
        for x_c, y_c, order in self.conditions:
            x_c_tf = tf.constant([x_c], dtype = tf.float32)
            with tf.GradientTape(persistent = True) as tape_c:
                tape_c.watch(x_c_tf)
                y_c_NN = self(x_c_tf, training = True)
                for _ in range(order):
                    y_c_NN = tape_c.gradient(y_c_NN, x_c_tf)
            loss += self.compiled_loss(y_c_NN, tf.constant([y_c], dtype = tf.float32))

        # Update weights
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Update metrics
        self.compiled_metrics.update_state(y_exact, self(x, training = True))
        return {m.name: m.result() for m in self.metrics}


"""
# Example usage for IVP: y'' + y = 0, y(0) = 1, y'(0) = 0
def ode_function_ivp(x, y, dy_dx, d2y_dx2):
    return d2y_dx2 + y

ivp_conditions = [(0.0, 1.0, 0), (0.0, 0.0, 1)]  # y(0) = 1, y'(0) = 0
input = tf.keras.Input(shape=(1,))
x = tf.keras.layers.Dense(50, activation='elu')(input)
x = tf.keras.layers.Dense(50, activation='elu')(x)
output = tf.keras.layers.Dense(1)(x)
model_ivp = ODE_nth(ode_function_ivp, ivp_conditions, n=2, condition_type='IVP')

# Compile and train IVP
x_train = np.linspace(0, 4, 20)
y_train = np.cos(x_train)  # Exact solution
model_ivp.compile(optimizer='adam', loss='mse', metrics=['mse'])
model_ivp.fit(x_train, y_train, batch_size=1, epochs=40)

# Example usage for BVP: y'' + y = 0, y(0) = 1, y(π/2) = 0
def ode_function_bvp(x, y, dy_dx, d2y_dx2):
    return d2y_dx2 + y

bvp_conditions = [(0.0, 1.0, 0), (np.pi/2, 0.0, 0)]  # y(0) = 1, y(π/2) = 0
model_bvp = ODE_nth(ode_function_bvp, bvp_conditions, n=2, condition_type='BVP')

# Compile and train BVP
x_train_bvp = np.linspace(0, np.pi/2, 20)
y_train_bvp = np.cos(x_train_bvp)  # Exact solution
model_bvp.compile(optimizer='adam', loss='mse', metrics=['mse'])
model_bvp.fit(x_train_bvp, y_train_bvp, batch_size=1, epochs=40)
"""

