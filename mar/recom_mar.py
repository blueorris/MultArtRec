import numpy as np
from tqdm.auto import trange

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import get_rng
from ...utils.init_utils import xavier_uniform


class MAR(Recommender):
    """Collaborative Deep Learning.

    Parameters
    ----------
    name: string, default: 'MAR'
        The name of the recommender model.

    input_size: int, default: None
        The dimension of the input.

    K: int, optional, default: 50
        The dimension of the latent topic layer and the latent feature layer.

    training_epochs: int, optional, default: 100
        Number of the training epochs.

    mlp_layers: list, default: None
        The number of neurons of encoder layer.
        For example, mlp_layers = [200], the MLP structure will be [input_size, 200, k]
    
    act_fn: str, default: 'relu'
        Name of the activation function used in the model structure.
        Supported functions: ['sigmoid', 'tanh', 'elu', 'relu', 'relu6', 'leaky_relu', 'identity']

    learning_rate: float, optional, default: 0.0001
        The learning rate for AdamOptimizer.

    lambda_n: float, optional, default: 1000
        The bias for reconstruction loss.
    
    lambda_r: float, optional, default: 1
        The bias for regularization loss.

    lambda_v: float, optional, default: 10
        The bias for item loss.

    lambda_c: float, optional, default: 1
        The bias for rating loss.

    lambda_t: float, optional, default: 0.1
        The bias for topic loss.

    lambda_w: float, optional, default: 0.1
        The parameter of model weights regularization.

    a: float, optional, default: 1
        The confidence of observed ratings.

    b: float, optional, default: 0.01
        The confidence of unseen ratings.

    batch_size: int, optional, default: 128
        The batch size for training model.
    
    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V}

        U: ndarray, shape (n_users,k)
            The user latent factors, optional initialization via init_params.
        V: ndarray, shape (n_items,k)
            The item latent factors, optional initialization via init_params.

    seed: int, optional, default: None
        Random seed for weight initialization.

    """

    def __init__(
        self,
        name="MultArtRec",
        input_size=None,
        K=50,
        mlp_layers=None,
        act_fn="relu",
        lambda_n=1000,
        lambda_r=1,
        lambda_v=10,
        lambda_c=1,
        lambda_t=0.1,
        lambda_w=0.1,
        a=1,
        b=0.01,
        learning_rate=0.0001,
        batch_size=128,
        training_epochs=100,
        verbose=True,
        init_params=None,
        seed=123,
    ):
        super().__init__(name=name, verbose=verbose)
        self.k = K
        self.lambda_n = lambda_n
        self.lambda_r = lambda_r
        self.lambda_v = lambda_v
        self.lambda_c = lambda_c
        self.lambda_t = lambda_t
        self.lambda_w = lambda_w
        self.a = a
        self.b = b
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.name = name
        self.training_epochs = training_epochs
        self.mlp_layers = mlp_layers
        self.act_fn = act_fn
        self.batch_size = batch_size
        self.verbose = verbose
        self.seed = seed
        self.rng = get_rng(seed)

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.U = self.init_params.get("U", None)
        self.V = self.init_params.get("V", None)

    def _init(self):
        n_users, n_items = self.train_set.num_users, self.train_set.num_items

        if self.U is None:
            self.U = xavier_uniform((n_users, self.k), self.rng)
        if self.V is None:
            self.V = xavier_uniform((n_items, self.k), self.rng)

    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)
        self._init()
        self._fit_mar()
        return self

    def _fit_mar(self):
        import tensorflow.compat.v1 as tf
        from .mar import Model
        
        tf.disable_eager_execution()

        R = self.train_set.csc_matrix
        n_users, n_items = self.train_set.num_users, self.train_set.num_items

        text_feature = self.train_set.item_text.batch_bow(
            np.arange(n_items)
        )
        text_feature = (text_feature - text_feature.min()) / (
            text_feature.max() - text_feature.min()
        )  # normalization

        # Build model
        mlp_layer_sizes = (
            [self.input_size]
            + self.mlp_layers
            + [self.k]
        )
        
        tf.set_random_seed(self.seed)
        model = Model(
            n_users=n_users,
            n_items=n_items,
            n_input=self.input_size,
            k=self.k,
            layers=mlp_layer_sizes,
            lambda_n=self.lambda_n,
            lambda_r=self.lambda_r,
            lambda_v=self.lambda_v,
            lambda_c=self.lambda_c,
            lambda_t=self.lambda_t,
            lambda_w=self.lambda_w,
            lr=self.learning_rate,
            U=self.U,
            V=self.V,
            act_fn=self.act_fn,
            seed=self.seed,
        )

        # Training model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            loop = trange(self.training_epochs, disable=not self.verbose)
            for _ in loop:
                sum_loss = 0
                count = 0
                for i, batch_ids in enumerate(
                    self.train_set.item_iter(self.batch_size, shuffle=True)
                ):
                    batch_R = R[:, batch_ids]
                    batch_C = np.ones(batch_R.shape) * self.b
                    batch_C[batch_R.nonzero()] = self.a

                    feed_dict = {
                        model.text_input: text_feature[batch_ids],
                        model.ratings: batch_R.A,
                        model.C: batch_C,
                        model.item_ids: batch_ids,
                    }
                    sess.run(model.opt1, feed_dict)  # train U, V
                    _, _loss = sess.run(
                        [model.opt2, model.loss], feed_dict
                    )

                    sum_loss += _loss
                    count += len(batch_ids)
                    if i % 10 == 0:
                        loop.set_postfix(loss=(sum_loss / count))

            self.U, self.V = sess.run([model.U, model.V])

        tf.reset_default_graph()

        if self.verbose:
            print("Learning completed!")

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items
        """
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            known_item_scores = self.V.dot(self.U[user_idx, :])
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            user_pred = self.V[item_idx, :].dot(self.U[user_idx, :])
            return user_pred
