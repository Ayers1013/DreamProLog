'''
This file was copied from https://github.com/zsoltzombori/mapping/blob/main/transformer/losses.py
'''

import tensorflow as tf

@tf.custom_gradient
def LogSumExp(x, axis, mask):
    y = tf.math.log(tf.reduce_sum(mask * tf.math.exp(x), axis=axis))
    y = tf.clip_by_value(y, -100, 100)
    # y = tf.math.reduce_logsumexp(x, axis=axis)

    def grad(upstream):
        x2 = x - tf.reduce_max(x, axis=axis)
        e_x = tf.exp(x2) * mask
        softmax = e_x / (1e-10 + tf.reduce_sum(e_x, axis=axis))
        softmax *= mask
        return upstream * softmax, tf.constant(0.0), tf.constant(0.0)

    return y, grad

def loss_function(real, pred, ispositive=False):
    mask_zero = tf.math.equal(real, 0)
    mask_nonzero = tf.math.logical_not(mask_zero)
    mask_zero = tf.cast(mask_zero, dtype=pred.dtype)
    mask_nonzero = tf.cast(mask_nonzero, dtype=pred.dtype)
    mask_nonzero_sequence = tf.reduce_max(mask_nonzero, axis=2)

    logprobs = pred - tf.math.reduce_logsumexp(pred, axis=-1, keepdims=True) #(support * bs * seq * tokens)
    
    # focus on the logprobs of real sequence
    logprobs = tf.gather(logprobs, real, batch_dims=3) #(support * bs * seq)
    # print("logprob", tf.transpose(logprobs, perm=[1,0,2])[0])

    # replace padding element probs with 1 for multiplication
    logprobs *= mask_nonzero
    sequence_logprobs = tf.reduce_sum(logprobs, axis=2) #(support * bs)
    # print("sequence_logprobs", tf.transpose(sequence_logprobs, perm=[1,0])[0])

    # reduce logprobs for all supporting sequences, removing padding sequences
    sequence_logprobs_all = LogSumExp(sequence_logprobs, 0, mask_nonzero_sequence)
    # print("sequence_logprobs_all", sequence_logprobs_all)

    sequence_probs_all = tf.math.exp(sequence_logprobs_all)
    # print("sequence_probs_all", sequence_probs_all)
    
    if ispositive:
        loss = - sequence_logprobs_all
        seq_weight = tf.stop_gradient(1 - sequence_probs_all)
        # loss = seq_weight * loss
    else:
        loss = tf.maximum(0.0, sequence_logprobs_all + 30.0)
        # seq_weight = tf.stop_gradient(sequence_probs_all)
    
    # sequence_logprobs_all = LogSumExp(sequence_logprobs)

    # print("\n______ sequence_probs_______")
    # print(tf.transpose(real, perm=[1, 0, 2]))
    
    # print(tf.transpose(logprobs, perm=[1, 0, 2, 3]))
    # print(tf.transpose(target_logprobs, perm=[1, 0, 2]))
    # print(tf.transpose(sequence_logprobs))
    # sequence_probs = tf.math.exp(sequence_logprobs)
    # print("PROBS: ", tf.transpose(sequence_probs))

    # loss = sequence_logprobs_all
    # loss = tf.exp(loss)

    loss = tf.reduce_mean(loss)
    probs = tf.reduce_mean(sequence_probs_all)
    return loss, probs
