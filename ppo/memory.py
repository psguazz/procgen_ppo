import tensorflow as tf
import numpy as np
from ppo.config import BATCHES, BATCH_SIZE


class Batch:
    def __init__(self, s_ts, a_ts, p_ts, v_ts, r_t1s, s_t1s, v_t1s, d_ts):
        self.s_ts = s_ts
        self.a_ts = a_ts
        self.p_ts = p_ts
        self.v_ts = v_ts
        self.r_t1s = r_t1s
        self.s_t1s = s_t1s
        self.v_t1s = v_t1s
        self.d_ts = d_ts


class Memory:
    def __init__(self):
        self.forget()

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size >= BATCH_SIZE * BATCHES

    def remember(self, s_t, a_t, p_t, v_t, r_t1, s_t1, v_t1, d_t):
        self.s_ts.append(s_t)
        self.a_ts.append(a_t)
        self.p_ts.append(p_t)
        self.v_ts.append(v_t)
        self.r_t1s.append(r_t1)
        self.s_t1s.append(s_t1)
        self.v_t1s.append(v_t1)
        self.d_ts.append(d_t)

        self.size += 1

    def forget(self):
        self.s_ts = []
        self.a_ts = []
        self.p_ts = []
        self.v_ts = []
        self.r_t1s = []
        self.s_t1s = []
        self.v_t1s = []
        self.d_ts = []

        self.size = 0

    def batches(self):
        s_ts = tf.convert_to_tensor(self.s_ts, dtype=tf.float32)
        a_ts = tf.convert_to_tensor(self.a_ts, dtype=tf.float32)
        p_ts = tf.convert_to_tensor(self.p_ts, dtype=tf.float32)
        v_ts = tf.convert_to_tensor(self.v_ts, dtype=tf.float32)
        r_t1s = tf.convert_to_tensor(self.r_t1s, dtype=tf.float32)
        s_t1s = tf.convert_to_tensor(self.s_t1s, dtype=tf.float32)
        v_t1s = tf.convert_to_tensor(self.v_t1s, dtype=tf.float32)
        d_ts = tf.convert_to_tensor(self.d_ts, dtype=tf.float32)

        indices = np.arange(self.size)
        np.random.shuffle(indices)

        starts = np.arange(0, self.size, BATCH_SIZE)

        for s in starts:
            batch = indices[s:s+BATCH_SIZE]

            yield Batch(
                s_ts=tf.gather(s_ts, batch),
                a_ts=tf.gather(a_ts, batch),
                p_ts=tf.gather(p_ts, batch),
                v_ts=tf.gather(v_ts, batch),
                r_t1s=tf.gather(r_t1s, batch),
                s_t1s=tf.gather(s_t1s, batch),
                v_t1s=tf.gather(v_t1s, batch),
                d_ts=tf.gather(d_ts, batch),
            )
