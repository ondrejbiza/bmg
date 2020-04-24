import numpy as np
import tensorflow as tf


class GMM_TF:

    def __init__(self, num_components, dimensionality):

        self.num_components = num_components
        self.dimensionality = dimensionality

    def fit(self, data, batch_size=100, num_steps=100, learning_rate=0.1, prior_entropy_weight=0.0):

        self.build_(learning_rate, prior_entropy_weight)
        self.start_session_()

        indices = np.array(list(range(len(data))), dtype=np.int32)
        lls = []

        for step_idx in range(num_steps):

            if step_idx > 0 and step_idx % 200 == 0:
                print("training step {:d}".format(step_idx))

            batch_indices = np.random.choice(indices, size=batch_size, replace=False)
            batch = data[batch_indices]
            _, ll = self.session.run([self.train_step, self.log_likelihood_t], feed_dict={
                self.input_pl: batch
            })
            lls.append(ll)

        return lls

    def sample(self, num):

        means, vars, prior = self.get_parameters()

        comps = np.random.choice(list(range(self.num_components)), size=num, replace=True, p=prior)

        samples = np.zeros((num, self.dimensionality), dtype=np.float32)

        for comp_idx in range(self.num_components):

            mask = comps == comp_idx

            if np.sum(mask) > 0:

                samples[mask] = np.random.multivariate_normal(
                    means[comp_idx], np.diag(np.sqrt(vars[comp_idx])), size=np.sum(mask)
                )

        return samples, comps

    def get_parameters(self):

        return self.session.run([self.means_v, self.vars_t, self.prior_cat_t])

    def build_(self, learning_rate, prior_entropy_weight):

        self.input_pl = tf.placeholder(tf.float32, (None, self.dimensionality), name="input_pl")

        self.means_v = tf.get_variable(
            "means_v", shape=(self.num_components, self.dimensionality), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=0.1)
        )

        self.sds_v = tf.get_variable(
            "sds_v", shape=(self.num_components, self.dimensionality), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=1, stddev=0.1)
        )
        self.sds_t = tf.nn.softplus(self.sds_v)
        self.vars_t = tf.square(self.sds_t)

        self.prior_v = tf.get_variable(
            "prior_v", shape=(self.num_components,), dtype=tf.float32,
            initializer=tf.constant_initializer(value=1.0)
        )
        self.prior_t = tf.nn.softplus(self.prior_v)
        self.prior_cat_t = self.prior_t / tf.reduce_sum(self.prior_t)
        self.prior_log_cat_t = tf.log(self.prior_cat_t + 1e-6)

        self.prior_entropy_t = - tf.reduce_sum(self.prior_cat_t * self.prior_log_cat_t)

        self.dist = tf.contrib.distributions.MultivariateNormalDiag(
            loc=self.means_v, scale_diag=self.sds_t
        )
        self.sample_log_probs_t = self.dist.log_prob(self.input_pl[:, tf.newaxis, :])

        self.full_log_likelihood_t = tf.reduce_logsumexp(
            self.sample_log_probs_t + self.prior_log_cat_t[tf.newaxis, :], axis=1
        )
        self.log_likelihood_t = tf.reduce_mean(self.full_log_likelihood_t) - \
            prior_entropy_weight * self.prior_entropy_t

        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_step = self.opt.minimize(- self.log_likelihood_t)

    def start_session_(self):

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
