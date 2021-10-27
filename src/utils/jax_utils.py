#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Sep 29 15:48:32 2020


@author: afsaneh

"""

import numpy as np
import pandas as pd
import functools
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import jax.numpy.linalg as jnla


# low_b=.001

@jax.jit
def modist(v):
    return jnp.median(v)


@jax.jit
def sum_jit(A, B):
    return jnp.sum(A, B)

@jax.jit
def Hadamard_prod(A, B):
    return A * B

@jax.jit
def kron_prod(a, b):
    return jnp.kron(a, b)


@jax.jit
def modif_kron(x, y):
    if (y.shape[1] != x.shape[1]):
        print("Column_number error")
    else:
        return jnp.array(list(jnp.kron(x[:, i], y[:, i]).T for i in list(range(y.shape[1]))))

@jax.jit
def stage2_weights(Gamma_w, Sigma_inv):
    n_row = Gamma_w.shape[0]
    arr = [mat_mul(jnp.diag(Gamma_w[i, :]), Sigma_inv) for i in range(n_row)]
    return jnp.concatenate(arr, axis=0)

@jax.jit
def mat_trans(A):
    return jnp.transpose(A)

@jax.jit
def mat_mul(A, B):
    return jnp.matmul(A, B)


@jax.jit
def cal_loocv_emb(K, kernel_y, lam):
    nD = K.shape[0]
    I = jnp.eye(nD)
    Q = jsla.inv(K + lam * nD * I)
    H = I - K.dot(Q)
    tildeH_inv = jnp.diag(1.0 / jnp.diag(H))
    return jnp.trace(tildeH_inv @ H @ kernel_y @ H @ tildeH_inv)


@jax.jit
def cal_loocv_alpha(K, sigma, gamma, y, lam):
    nD = K.shape[0]
    I = jnp.eye(nD)
    H = I - mat_mul(mat_mul(K, gamma), (jsla.inv(sigma + lam * nD * I)))
    tildeH_inv = jnp.diag(1.0 / jnp.diag(H))
    return jnp.linalg.norm(tildeH_inv.dot(H.dot(y)))
