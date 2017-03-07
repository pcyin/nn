# -*- coding: utf-8 -*-

import logging
import theano
import theano.tensor as T
import numpy as np

from .core import *


def get_mask(mask, X):
    if mask is None:
        mask = T.ones((X.shape[1], X.shape[0], 1), dtype='float32')  # (time, nb_samples, 1)
    else:
        mask = T.shape_padright(mask)
        mask = mask.dimshuffle((1, 0, 2))
    return mask


class GRU(Layer):
    '''
        Gated Recurrent Unit - Cho et al. 2014

        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
                http://www.aclweb.org/anthology/W14-4012
            Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
                http://arxiv.org/pdf/1412.3555v1.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 return_sequences=False, name='GRU'):

        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((self.input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if name is not None:
            self.set_name(name)

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        # h_tm1 = theano.printing.Print(self.name + 'h_tm1::')(h_tm1)
        h_mask_tm1 = mask_tm1 * h_tm1
        # h_mask_tm1 = theano.printing.Print(self.name + 'h_mask_tm1::')(h_mask_tm1)
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = z * h_mask_tm1 + (1 - z) * hh_t
        return h_t

    def __call__(self, X, mask=None, init_state=None):
        padded_mask = self.get_padded_shuffled_mask(mask, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h

        if init_state:
            # (batch_size, output_dim)
            outputs_info = T.unbroadcast(init_state, 1)
        else:
            outputs_info = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        outputs, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=outputs_info,
            non_sequences=[self.U_z, self.U_r, self.U_h])

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_padded_shuffled_mask(self, mask, X, pad=0):
        # mask is (nb_samples, time)
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8')


class GRU_4BiRNN(Layer):
    '''
        Gated Recurrent Unit - Cho et al. 2014

        Acts as a spatiotemporal projection,
        turning a sequence of vectors into a single vector.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
                http://www.aclweb.org/anthology/W14-4012
            Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
                http://arxiv.org/pdf/1412.3555v1.pdf
    '''
    def __init__(self, input_dim, output_dim=128,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 return_sequences=False, name=None):

        super(GRU_4BiRNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.W_z = self.init((self.input_dim, self.output_dim))
        self.U_z = self.inner_init((self.output_dim, self.output_dim))
        self.b_z = shared_zeros((self.output_dim))

        self.W_r = self.init((self.input_dim, self.output_dim))
        self.U_r = self.inner_init((self.output_dim, self.output_dim))
        self.b_r = shared_zeros((self.output_dim))

        self.W_h = self.init((self.input_dim, self.output_dim))
        self.U_h = self.inner_init((self.output_dim, self.output_dim))
        self.b_h = shared_zeros((self.output_dim))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        if name is not None:
            self.set_name(name)

    def _step(self,
              # xz_t, xr_t, xh_t, mask_tm1, mask,
              xz_t, xr_t, xh_t, mask,
              h_tm1,
              u_z, u_r, u_h):
        # h_mask_tm1 = mask_tm1 * h_tm1
        # h_tm1 = theano.printing.Print(self.name + '::h_tm1::')(h_tm1)
        # mask = theano.printing.Print(self.name + '::mask::')(mask)

        z = self.inner_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * hh_t

        # mask
        h_t = (1 - mask) * h_tm1 + mask * h_t
        # h_t = theano.printing.Print(self.name + '::h_t::')(h_t)

        return h_t

    def __call__(self, X, mask=None, init_state=None):
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)
        mask = mask.astype('int8')
        # mask, padded_mask = self.get_padded_shuffled_mask(mask, pad=1)
        X = X.dimshuffle((1, 0, 2))

        x_z = T.dot(X, self.W_z) + self.b_z
        x_r = T.dot(X, self.W_r) + self.b_r
        x_h = T.dot(X, self.W_h) + self.b_h

        if init_state:
            # (batch_size, output_dim)
            outputs_info = T.unbroadcast(init_state, 1)
        else:
            outputs_info = T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)

        outputs, updates = theano.scan(
            self._step,
            # sequences=[x_z, x_r, x_h, padded_mask, mask],
            sequences=[x_z, x_r, x_h, mask],
            outputs_info=outputs_info,
            non_sequences=[self.U_z, self.U_r, self.U_h])

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def get_padded_shuffled_mask(self, mask, pad=0):
        assert mask, 'mask cannot be None'
        # mask is (nb_samples, time)
        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            padded_mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8'), padded_mask.astype('int8')


class LSTM(Layer):
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='sigmoid', name='LSTM'):

        super(LSTM, self).__init__()

        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.input_dim = input_dim

        self.W_i = self.init((input_dim, self.output_dim))
        self.U_i = self.inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim))

        self.W_f = self.init((input_dim, self.output_dim))
        self.U_f = self.inner_init((self.output_dim, self.output_dim))
        self.b_f = self.forget_bias_init((self.output_dim))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

        self.set_name(name)

    def _step(self,
              x_stacked_t, mask_t,
              h_tm1, c_tm1,
              u_stacked):

        # (batch_size, output_dim * 4)
        gates = x_stacked_t + T.dot(h_tm1, u_stacked)

        # Extract the pre-activation gate values and apply non-linearaities
        i_t = self.inner_activation(gates[:, 0: self.output_dim])
        f_t = self.inner_activation(gates[:, self.output_dim: 2 * self.output_dim])
        cell_input = self.activation(gates[:, 2 * self.output_dim: 3 * self.output_dim])
        o_t = self.inner_activation(gates[:, 3 * self.output_dim: 4 * self.output_dim])

        # Compute new cell value
        c_t = f_t * c_tm1 + i_t * cell_input
        h_t = o_t * self.activation(c_t)

        h_t = (1. - mask_t) * h_tm1 + mask_t * h_t
        c_t = (1. - mask_t) * c_tm1 + mask_t * c_t

        return h_t, c_t

    def __call__(self, X, mask=None, init_state=None, init_cell=None, one_step=False, return_sequences=True):
        mask = get_mask(mask, X)
        batch_size = X.shape[0]
        X = X.dimshuffle((1, 0, 2))

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        # code from: https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/recurrent.py

        W_in_stacked = T.concatenate([self.W_i, self.W_f, self.W_c, self.W_o], axis=-1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate([self.U_i, self.U_f, self.U_c, self.U_o], axis=-1)

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate([self.b_i, self.b_f, self.b_c, self.b_o], axis=0)

        # pre compute input transformations
        X_trans = T.dot(X, W_in_stacked) + b_stacked

        if not init_state:
            init_state = T.alloc(0., batch_size, self.output_dim)
        if not init_cell:
            init_cell = T.alloc(0., batch_size, self.output_dim)

        sequences = [X_trans, mask]
        outputs_info = [init_state, init_cell]
        non_sequences = [W_hid_stacked]

        if one_step:
            states, cells = _step(*(sequences + outputs_info + non_sequences))
        else:
            [states, cells], updates = theano.scan(
                self._step,
                sequences=sequences,
                outputs_info=outputs_info,
                non_sequences=non_sequences)

        if return_sequences:
            return states.dimshuffle((1, 0, 2)), cells.dimshuffle((1, 0, 2))
        else:
            return states[-1], cells[-1]


class BiLSTM(Layer):
    def __init__(self, input_dim, output_dim,
                 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
                 activation='tanh', inner_activation='sigmoid', return_sequences=False, name='BiLSTM'):
        super(BiLSTM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.return_sequences = return_sequences

        params = dict(locals())
        del params['self']

        params['name'] = 'foward_lstm'
        self.forward_lstm = LSTM(**params)
        params['name'] = 'backward_lstm'
        self.backward_lstm = LSTM(**params)

        self.params = self.forward_lstm.params + self.backward_lstm.params

        self.set_name(name)

    def __call__(self, X, mask=None, init_state=None, dropout=0, train=True, srng=None):
        # X: (nb_samples, nb_time_steps, embed_dim)
        # mask: (nb_samples, nb_time_steps)
        if mask is None:
            mask = T.ones((X.shape[0], X.shape[1]))

        hidden_states_forward = self.forward_lstm(X, mask, init_state, dropout, train, srng)
        hidden_states_backward = self.backward_lstm(X[:, ::-1, :], mask[:, ::-1], init_state, dropout, train, srng)

        if self.return_sequences:
            hidden_states = T.concatenate([hidden_states_forward, hidden_states_backward[:, ::-1, :]], axis=-1)
        else:
            raise NotImplementedError()

        return hidden_states


class GRUDecoder(Layer):
    '''
        GRU Decoder
    '''
    def __init__(self, input_dim, context_dim, hidden_dim, vocab_num,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', inner_activation='sigmoid',
                 name='GRUDecoder'):

        super(GRUDecoder, self).__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.vocab_num = vocab_num

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        self.W_z = self.init((self.input_dim, self.hidden_dim))
        self.U_z = self.inner_init((self.hidden_dim, self.hidden_dim))
        self.C_z = self.init((self.context_dim, self.hidden_dim))
        self.b_z = shared_zeros((self.hidden_dim))

        self.W_r = self.init((self.input_dim, self.hidden_dim))
        self.U_r = self.inner_init((self.hidden_dim, self.hidden_dim))
        self.C_r = self.init((self.context_dim, self.hidden_dim))
        self.b_r = shared_zeros((self.hidden_dim))

        self.W_h = self.init((self.input_dim, self.hidden_dim))
        self.U_h = self.inner_init((self.hidden_dim, self.hidden_dim))
        self.C_h = self.init((self.context_dim, self.hidden_dim))
        self.b_h = shared_zeros((self.hidden_dim))

        # self.W_y = self.init((self.input_dim, self.vocab_num))
        self.U_y = self.init((self.hidden_dim, self.vocab_num))
        self.C_y = self.init((self.context_dim, self.vocab_num))
        self.b_y = shared_zeros((self.vocab_num))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
            self.C_z, self.C_r, self.C_h,
            self.U_y, self.C_y, self.b_y, #self.W_y
        ]

        if name is not None:
            self.set_name(name)

    def _step(self,
              xz_t, xr_t, xh_t, mask_tm1,
              h_tm1,
              u_z, u_r, u_h):
        h_mask_tm1 = mask_tm1 * h_tm1
        z = self.inner_activation(xz_t + T.dot(h_mask_tm1, u_z))
        r = self.inner_activation(xr_t + T.dot(h_mask_tm1, u_r))
        hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1, u_h))
        h_t = z * h_mask_tm1 + (1 - z) * hh_t
        return h_t

    def __call__(self, target, context, mask=None):
        target = target * T.cast(T.shape_padright(mask), 'float32')
        padded_mask = self.get_padded_shuffled_mask(mask, pad=1)
        # target = theano.printing.Print('X::' + self.name)(target)
        X_shifted = T.concatenate([alloc_zeros_matrix(target.shape[0], 1, self.input_dim), target[:, 0:-1, :]], axis=-2)

        # X = theano.printing.Print('X::' + self.name)(X)
        # X = T.zeros_like(target)
        # T.set_subtensor(X[:, 1:, :], target[:, 0:-1, :])

        X = X_shifted.dimshuffle((1, 0, 2))

        ctx_step = context.dimshuffle(('x', 0, 1))
        x_z = T.dot(X, self.W_z) + T.dot(ctx_step, self.C_z) + self.b_z
        x_r = T.dot(X, self.W_r) + T.dot(ctx_step, self.C_r) + self.b_r
        x_h = T.dot(X, self.W_h) + T.dot(ctx_step, self.C_h) + self.b_h

        h, updates = theano.scan(
            self._step,
            sequences=[x_z, x_r, x_h, padded_mask],
            outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.hidden_dim), 1),
            non_sequences=[self.U_z, self.U_r, self.U_h])

        # (batch_size, max_token_len, hidden_dim)
        h = h.dimshuffle((1, 0, 2))

        # (batch_size, max_token_len, vocab_size)
        predicts = T.dot(h, self.U_y) + T.dot(context.dimshuffle((0, 'x', 1)), self.C_y) + self.b_y # + T.dot(X_shifted, self.W_y)

        predicts_flatten = predicts.reshape((-1, predicts.shape[2]))
        return T.nnet.softmax(predicts_flatten).reshape((predicts.shape[0], predicts.shape[1], predicts.shape[2]))

    def get_padded_shuffled_mask(self, mask, pad=0):
        assert mask, 'mask cannot be None'
        # mask is (nb_samples, time)
        mask = T.shape_padright(mask)  # (nb_samples, time, 1)
        mask = T.addbroadcast(mask, -1)  # (time, nb_samples, 1) matrix.
        mask = mask.dimshuffle(1, 0, 2)  # (time, nb_samples, 1)

        if pad > 0:
            # left-pad in time with 0
            padding = alloc_zeros_matrix(pad, mask.shape[1], 1)
            mask = T.concatenate([padding, mask], axis=0)
        return mask.astype('int8')
