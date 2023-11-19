import tensorflow.compat.v1 as tf

from ...utils import get_rng
from ...utils.init_utils import xavier_uniform

act_functions = {
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh,
    "elu": tf.nn.elu,
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
    "leaky_relu": tf.nn.leaky_relu,
    "identity": tf.identity,
}

class Model:
    def __init__(
        self,
        n_users,
        n_items,
        n_input,
        k,
        layers,
        lambda_n,
        lambda_r,
        lambda_v,
        lambda_c,
        lambda_t,
        lambda_w,
        lr,
        U,
        V,
        act_fn,
        seed,
    ):
        self.input_dim = n_input
        self.n_z = k
        self.n_users = n_users
        self.n_items = n_items
        self.lambda_n = lambda_n 
        self.lambda_r = lambda_r
        self.lambda_v = lambda_v
        self.lambda_c = lambda_c
        self.lambda_t = lambda_t
        self.lambda_w = lambda_w
        self.layers = layers
        self.lr = lr
        self.k = k
        self.U_init = tf.constant(U)
        self.V_init = tf.constant(V)
        self.act_fn = act_fn
        self.seed = seed

        self._build_graph()

    def _build_graph(self):
        self.text_input = tf.placeholder(
            dtype=tf.float32, shape=[None, self.input_dim], name="text_input"
        )
        self.ratings = tf.placeholder(
            dtype=tf.float32, shape=[self.n_users, None], name="rating_input"
        )
        self.C = tf.placeholder(
            dtype=tf.float32, shape=[self.n_users, None], name="C_input"
        )

        with tf.variable_scope("CF_Variable"):
            self.U = tf.get_variable(
                name="U", dtype=tf.float32, initializer=self.U_init
            )
            self.V = tf.get_variable(
                name="V", dtype=tf.float32, initializer=self.V_init
            )

        self.item_ids = tf.placeholder(dtype=tf.int32)
        real_batch_size = tf.cast(tf.shape(self.text_input)[0], tf.int32)
        V_batch = tf.reshape(
            tf.gather(self.V, self.item_ids), shape=[real_batch_size, self.k]
        )

        topic_embd, x_recon, reg_loss = self._vae(self.text_input)

        loss_1 = self.lambda_w * (tf.nn.l2_loss(self.U) + reg_loss)  # weights regularization
        loss_2 = self.lambda_n * tf.nn.l2_loss(x_recon - self.text_input)  # reconstruction loss
        loss_3 = self.lambda_v * tf.nn.l2_loss(V_batch - topic_embd)  # item loss

        predictions = tf.matmul(self.U, V_batch, transpose_b=True)
        squared_error = tf.square(self.ratings - predictions)
        loss_4 = self.lambda_c * tf.reduce_sum(tf.multiply(self.C, squared_error))  # rating loss

        loss_5 = self.lambda_r * 0.5 * tf.reduce_mean(
            tf.reduce_sum(
                tf.square(self.h_mean)
                + tf.exp(self.h_log_sigma_sq)
                - self.h_log_sigma_sq
                - 1,
                1,
            )
        )  # regularization loss

        loss_6 = self.lambda_t * tf.reduce_mean(tf.square(tf.subtract(self.h, self.z)))  # topic loss

        self.loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6

        # Generate optimizer
        optimizer1 = tf.train.AdamOptimizer(self.lr)
        optimizer2 = tf.train.AdamOptimizer(self.lr)

        train_var_list1, train_var_list2 = [], []

        for var in tf.trainable_variables():
            if "CF_Variable" in var.name:
                train_var_list1.append(var)
            elif "NTM_Variable" in var.name:
                train_var_list2.append(var)

        gvs = optimizer1.compute_gradients(self.loss, var_list=train_var_list1)
        capped_gvs = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gvs]
        self.opt1 = optimizer1.apply_gradients(capped_gvs)

        gvs = optimizer2.compute_gradients(self.loss, var_list=train_var_list2)
        capped_gvs = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gvs]
        self.opt2 = optimizer2.apply_gradients(capped_gvs)


    def _vae(self, X):
        act_fn = act_functions.get(self.act_fn, None)
        if act_fn is None:
            raise ValueError(
                "Invalid type of activation function {}\n"
                "Supported functions: {}".format(act_fn, act_functions.keys())
            )

        with tf.variable_scope("NTM_Variable_inference"):
            rec = {
                "W1": tf.get_variable(
                    "W1",
                    [self.input_dim, self.layers[0]],
                    initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
                    dtype=tf.float32,
                ),
                "b1": tf.get_variable(
                    "b1",
                    [self.layers[0]],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32,
                ),
                "W2": tf.get_variable(
                    "W2",
                    [self.layers[0], self.layers[1]],
                    initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
                    dtype=tf.float32,
                ),
                "b2": tf.get_variable(
                    "b2",
                    [self.layers[1]],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32,
                ),
                "W_h_mean": tf.get_variable(
                    "W_h_mean",
                    [self.layers[1], self.n_z],
                    initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
                    dtype=tf.float32,
                ),
                "b_h_mean": tf.get_variable(
                    "b_h_mean",
                    [self.n_z],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32,
                ),
                "W_z_log_sigma": tf.get_variable(
                    "W_z_log_sigma",
                    [self.layers[1], self.n_z],
                    initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
                    dtype=tf.float32,
                ),
                "b_z_log_sigma": tf.get_variable(
                    "b_z_log_sigma",
                    [self.n_z], # 50
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32,
                ),
                "W_h": tf.get_variable(
                    "W_h",
                    [self.n_z, self.n_z],
                    initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
                    dtype=tf.float32,
                ),
                "b_h": tf.get_variable(
                    "b_h",
                    [self.n_z],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32,
                ),
            }

        h_value = act_fn(tf.matmul(X, rec["W1"]) + rec["b1"])
        h_value = act_fn(tf.matmul(h_value, rec["W2"]) + rec["b2"])

        self.h_mean = tf.matmul(h_value, rec["W_h_mean"]) + rec["b_h_mean"]
        self.h_log_sigma_sq = tf.matmul(h_value, rec["W_z_log_sigma"]) + rec["b_z_log_sigma"]

        eps = tf.random_normal(
            shape=tf.shape(self.h_mean),
            mean=0,
            stddev=1,
            seed=self.seed,
            dtype=tf.float32,
        )

        self.h = (
            self.h_mean + tf.sqrt(tf.maximum(tf.exp(self.h_log_sigma_sq), 1e-10)) * eps
        )

        self.z = tf.matmul(self.h, rec["W_h"]) + rec["b_h"]

        with tf.variable_scope("NTM_Variable_generation"):
            gen = {
                "W1": tf.get_variable(
                    "W1",
                    [self.n_z, self.layers[1]],
                    initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
                    dtype=tf.float32,
                ),
                "b1": tf.get_variable(
                    "b1",
                    [self.layers[1]],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32,
                ),
                "W2": tf.get_variable(
                    "W2",
                    [self.layers[1], self.layers[0]],
                    initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
                    dtype=tf.float32,
                ),
                "b2": tf.get_variable(
                    "b2",
                    [self.layers[0]],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32,
                ),
                "W_x": tf.get_variable(
                    "W_x",
                    [self.layers[0], self.input_dim],
                    initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
                    dtype=tf.float32,
                ),
                "b_x": tf.get_variable(
                    "b_x",
                    [self.input_dim],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32,
                ),
            }

        reg_loss = tf.constant(0.0)
        for i in rec:
            reg_loss = tf.add(reg_loss, tf.nn.l2_loss(rec[i]))
        for i in gen:
            reg_loss = tf.add(reg_loss, tf.nn.l2_loss(gen[i]))

        h_value = act_fn(tf.matmul(self.z, gen["W1"]) + gen["b1"])
        h_value = act_fn(tf.matmul(h_value, gen["W2"]) + gen["b2"])
        x_recon = tf.matmul(h_value, gen["W_x"]) + gen["b_x"]

        return self.z, x_recon, reg_loss
