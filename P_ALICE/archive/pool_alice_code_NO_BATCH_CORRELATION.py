import time
import numpy as np
from typing import List, Tuple
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats

def pool_alice(X_unlabeled: np.ndarray, oracle: np.ndarray, basis_funcs: List[np.ufunc], n_tr: int, prng_seed:int, passive_learning:bool=False)->np.ndarray: # type: ignore
    """
    Pseudocode from Fig 3. of P-ALICE (pool ALICE) paper, pg 256
    Paper Link: https://link.springer.com/article/10.1007/s10994-009-5100-3

    Args:
    - X_unlabeled: (n_te x t) feature matrix (pool of unlabeled samples)
    - oracle: a vector or function that given some index i from the pool X_unlabeled, returns the label of X_unlabeled[i]
    - basis_funcs: (t x 1) list of numpy functions that apply to the columns of X_unlabeled. the functions which when linearly combined form the (nonlinear) regression model we are learning. the t'th basis function, does some nonlinear/linear transformation on the t'th value of a sample (row from X).
    - n_tr: the number of points we are allowed to ask the oracle to label (NOTE, ASSUMED: n_tr << n_te)
    - prng_seed: int, seed for numpy random number generator
    
    Returns:
    - theta_hat_w: (t x 1)
    """

    if prng_seed == None:
        raise Exception('must set seed outside of funciton call for reproducibility')

    n_te = len(X_unlabeled) # number of points in unlabeled set
    te_idxs = [i for i in range(0,n_te)]

    # compute (t x t) matrix Uhat, using U_hat(l,l') formula
    U_hat, ϕ_mtx = compute_U_hat_and_ϕ_mtx(X_unlabeled,basis_funcs)

    if not passive_learning:
        # see eqns 78-80
        λ_vals: List[float] = [i for i in np.arange(0,1.1,0.1)] + [i for i in np.arange(0.40,0.61,0.01)]
    else:
        λ_vals: List[float] = [0.0]

    # compute (n_te x 1) vector b_λ(x) with λ=1 (to speed up computing b_λ in for loop)
    base_b_set: np.ndarray = compute_base_b_set_v2(U_hat,ϕ_mtx)

    # # list where the i'th value corresponds to the P-ALICE(λ) score of the i'th λ
    # P_ALICE_λ_all: List[float] = [0]*len(λ_vals)
    # tr_idxs_λ_all: List[np.ndarray] = [0]*len(λ_vals) # type: ignore
    # L_λ_all: List[np.ndarray] = [0]*len(λ_vals) # type: ignore
    max_P_ALICE_λ: float = float('-inf')
    best_idxs: np.ndarray
    best_L_λ: np.ndarray
    best_λ_val: float

    # declare variables
    b_λ_set:np.ndarray
    proba_b_λ:np.ndarray
    tr_idxs:np.ndarray
    # X_λ_tr:np.ndarray # during implementation, I realized that this is simply subsetting the ϕ_mtx
    X_λ:np.ndarray
    W_λ:np.ndarray
    L_λ:np.ndarray

    # for several different values of λ (possibly around 1/2)
    for λ_val in λ_vals:
        # print(f"λ_val={λ_val}")
        # compute resampling bias function under current λ i.e.
        # Compute {b_λ(x_te_j)} for j in [1,n_te]
        b_λ_set = compute_b_λ_set(base_b_set,λ_val)

        # probability we choose x from x_te under current λ
        proba_b_λ = b_λ_set/np.sum(b_λ_set)
        tr_idxs = np.random.choice(a=te_idxs,size=n_tr,replace=False,p=proba_b_λ)

        # BELOW is equivalent to simply subsetting ϕ_mtx 
        # # (n_tr x t) UNMAPPED condidate training set under current λ (not plugged into each basis function)
        # X_λ_tr = X_unlabeled[tr_idxs]
        # # (n_tr x t) MAPPED condidate training set under current λ
        # X_λ = compute_X_λ(X_λ_tr,basis_funcs)
        X_λ = ϕ_mtx[tr_idxs]

        # (n_tr x n_tr) diagonal weight matrix
        W_λ = np.diag(1/b_λ_set[tr_idxs],k=0)

        # compute L_λ (attempt exact inversion, then resort to pseudo)
        # print(f"X_λ.shape={X_λ.shape}, W_λ.shape={W_λ.shape}")
        try:
            L_λ = np.linalg.inv(X_λ.T @ W_λ @ X_λ) @ X_λ.T @ W_λ
        except:
            L_λ = np.linalg.pinv(X_λ.T @ W_λ @ X_λ) @ X_λ.T @ W_λ

        P_ALICE_λ = np.trace(U_hat @ L_λ @ L_λ.T)

        # save the best P_ALICE_λ and corresponding info so far
        if max_P_ALICE_λ < P_ALICE_λ:
            max_P_ALICE_λ = P_ALICE_λ
            best_idxs = tr_idxs
            best_L_λ = L_λ
            best_λ_val = λ_val
        
    # print(f"best_λ_val={best_λ_val}")
    # argmin_P_ALICE_idx = np.argmin(P_ALICE_λ)
    y_tr_labels = get_labels(oracle,best_idxs)

    # compute weight parameters
    theta_W = best_L_λ @ y_tr_labels

    return theta_W

def get_labels(oracle,best_idxs):
    """ in the "toy" pool based setting, we have all the labels, so
    if oracle equals y_unlabeled (the labels for the set X_unlabeled) then we can just numpy fancy index the oracle (y_labels) to get the labels
    """
    return oracle[best_idxs]
    
def compute_b_λ_set(base_b_set,λ_val)->np.ndarray:
    return base_b_set**λ_val


# # BAD version
# def compute_base_b_set_v1(U_hat:np.ndarray,ϕ_mtx:np.ndarray)->np.ndarray:

#     # creates a (n_tr x n_tr) matrix
#     try:
#         base_b_mtx = ϕ_mtx @ np.linalg.inv(U_hat) @ ϕ_mtx.T
#     except:
#         base_b_mtx = ϕ_mtx @ np.linalg.pinv(U_hat) @ ϕ_mtx.T

#     # sum over the columns to reduce to (n_tr x 1) vector
#     base_b_set = np.sum(base_b_mtx,axis=0)

#     assert all(base_b_set > 0)
#     return base_b_set

def compute_base_b_set_v2(U_hat:np.ndarray,ϕ_mtx:np.ndarray)->np.ndarray:
    """ Compute the resampling bias of each sample
    v2 is corrected compute_base_b_set. a little slower than v1, but correct.
    """

    # t = len(ϕ_mtx[0])

    # creates a (n_tr x n_tr) matrix
    U_hat_inv: np.ndarray
    try:
        U_hat_inv = np.linalg.inv(U_hat)
        # print("np.linalg.inv happened (not pseudo)")
    except:
        U_hat_inv = np.linalg.pinv(U_hat)
        # print("np.linalg.pinv happened (yes pesudo)")

    # base_b_set: np.ndarray = np.empty(shape=(1,t))
    # for row_idx, ϕ_vec in enumerate(ϕ_mtx):
    #     base_b_set[row_idx] = ϕ_vec @ U_hat_inv @ ϕ_vec.T

    base_b_set: np.ndarray = np.array([ϕ_vec @ U_hat_inv @ ϕ_vec.T for ϕ_vec in ϕ_mtx])

    # print(f"base_b_set.shape={base_b_set.shape}")
    # print("NEGATIVE ELEMENTS FROM U_hat:",U_hat[U_hat < 0])
    # print("NEGATIVE ELEMENTS FROM U_hat_inv:",U_hat_inv[U_hat_inv < 0],len(U_hat_inv[U_hat_inv < 0]))
    # np.save('base_b_set_debug_v2.npy',base_b_set)

    assert np.all((np.greater(base_b_set,0))), "base_b_set[i] > 0 is NOT true for all i (i.e. for every sample)"
    return base_b_set

# # v3
# def compute_base_b_set_v3(U_hat:np.ndarray,ϕ_mtx:np.ndarray)->np.ndarray:
    
#     t = len(ϕ_mtx[0])

#     # creates a (n_tr x n_tr) matrix
#     U_hat_inv: np.ndarray
#     try:
#         U_hat_inv = np.linalg.inv(U_hat)
#         print("np.linalg.inv happened (not pseudo)")
#     except:
#         U_hat_inv = np.linalg.pinv(U_hat)
#         print("np.linalg.pinv happened (yes pesudo)")

#     base_b_set: np.ndarray = np.zeros(shape=(1,len(ϕ_mtx)))

#     for phi_row in ϕ_mtx:
#         b_val = 0
#         for l_idx in range(t):
#             for l_prime_idx in range(t):
#                 b_val += U_hat_inv[l_idx,l_prime_idx] * phi_row[l_idx] * phi_row[l_prime_idx]

#     print(f"base_b_set.shape={base_b_set.shape}")
#     print("NEGATIVE ELEMENTS FROM U_hat:",U_hat[U_hat < 0])
#     print("NEGATIVE ELEMENTS FROM U_hat_inv:",U_hat_inv[U_hat_inv < 0],len(U_hat_inv[U_hat_inv < 0]))

#     np.save('base_b_set_debug_v2.npy',base_b_set)

#     assert np.all((np.greater(base_b_set,0))), "base_b_set > 0 is NOT true for all base_b_set[i]"
#     raise Exception('PAUSE')
#     return base_b_set


def compute_U_hat_and_ϕ_mtx(X_unlabeled:np.ndarray,basis_funcs:List[np.ufunc])->Tuple[np.ndarray,np.ndarray]:
    # t = len(X_unlabeled[0])
    # n_te = len(X_unlabeled)
    n_te, t = X_unlabeled.shape
    U_hat: np.ndarray = np.empty(shape=(t,t)) # np.full(shape=(t,t),fill_value=np.nan)
    ϕ_mtx: np.ndarray = np.empty(shape=(n_te,t)) # np.full(shape=(n_te,t),fill_value=np.nan) 

    # apply l'th basis function to l'th column of feature matrix
    for l in range(0,t):
        ϕ_mtx[:,l] = basis_funcs[l](X_unlabeled[:,l])

    U_hat = (1/n_te) * ϕ_mtx.T @ ϕ_mtx

    return U_hat, ϕ_mtx
    # the really slow way
    # for l in range(0,t):
    #     for l_prime in range(l,t):
    #         temp = (1/n_te)*np.sum(ϕ_mtx[:,l]*ϕ_mtx[:,l_prime])
    #         U_hat[l][l_prime]
    #         U_hat[l_prime][l]


#### BASIS FUNCTIONS  BEGIN ####
def identity_basis(x:np.ndarray):
    """ use this function to achieve pure linear regression, see equation
    3 of text. if we make phi(x) = x, then becomes normal linear reg. (multivariate)
    """
    return x # x*1

def squared_basis(x:np.ndarray):
    return x ** 2

def cubed_basis(x:np.ndarray):
    return x ** 3

def map_to_off_on_basis(x:np.ndarray):
    """ maps one hot encoded data such that
    0 is -1 and 1 is +1
    """
    return x*2 - 1
#### BASIS FUNCTIONS END ####

def predict(X_transformed:np.ndarray, weights:np.ndarray)->np.ndarray:
    """ X_transformed is the array formed from taking X and adding the additional columns (features) to make it X_pretransformed, then applying the collection of basis functions to each column of X_pretransformed"""

    if len(X_transformed[0]) != len(weights):
        error_string = f"""X_transformed must have same number of features as the number of weight parameters formed: len(X_transformed[0])={len(X_transformed[0])}, len(weights)={len(weights)}\n"""
        raise Exception(error_string)
    y_pred = X_transformed @ weights

    return y_pred

def apply_basis_funcs(X_pretransformed:np.ndarray, basis_funcs:List[np.ufunc])->np.ndarray:

    _, t = X_pretransformed.shape
    # print(f"X_pretransformed.shape[1] = t = {X_pretransformed.shape[1]}")
    if len(X_pretransformed[0]) != len(basis_funcs):
        error_string =\
f"""X_pretransformed must have same number of features as basis functions: len(X_pretransformed[0])={len(X_pretransformed[0])}, len(basis_funcs)={len(basis_funcs)}

Be careful to add additional columns to X to make X_pretransformed if using higher order features or nonlinear duplications of
features e.g. feature_1 + feature_1^2 + sin(feature_2) would require 3 columns: column 1 and column 2 are feature 1, and column 3
is feature 2"""
        raise Exception(error_string)

    X_transformed: np.ndarray = np.empty(shape=X_pretransformed.shape)
    # apply l'th basis function to l'th column of feature matrix
    for l in range(0,t):
        # print(f"basis_funcs[l]={basis_funcs[l]}")
        X_transformed[:,l] = basis_funcs[l](X_pretransformed[:,l])
    
    return X_transformed

def simulate_pool_alice_n_times(n: int, seeds: List[int], X_pool: np.ndarray, y_pool: np.ndarray, basis_funcs:List[np.ufunc],passive_learning:bool=False, batch_size: int = 1)->np.ndarray:

    sim_results: List[np.ndarray] = [None]*n # type: ignore
    
    if seeds == None or len(seeds) != n:
        raise Exception("Must supply n seeds")
    original_start_time = time.time()

    for i in range(0,n):
        prng_seed = seeds[i]

        if prng_seed == None:
            raise Exception('must provide a seed for reproducibility')
        else:
            np.random.seed(prng_seed)

        X_train, X_test, y_train, y_test = train_test_split(X_pool, y_pool, test_size=0.2, random_state=prng_seed, shuffle=True)
        X_test_transformed = apply_basis_funcs(X_test,basis_funcs)
        
        start_time = time.time()
        print(f"\n______________________________________________\nsimulation number={i+1}, prng_seed={prng_seed},start_time={start_time}")
        sim_results[i] = simulate_pool_alice_once(X_train=X_train,y_train=y_train, X_test_transformed=X_test_transformed,y_test=y_test,basis_funcs=basis_funcs,prng_seed=prng_seed,passive_learning=passive_learning, batch_size=batch_size,start_time=start_time)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed Time:", elapsed_time, "seconds")
        # raise Exception('PAUSE')
    elapsed_time = end_time - original_start_time
    print(f"TOTAL Elapsed Time: {elapsed_time} seconds ({elapsed_time/60} min)")
        
    return np.array(sim_results)

def simulate_pool_alice_once(X_train: np.ndarray, y_train: np.ndarray, X_test_transformed: np.ndarray, y_test: np.ndarray, basis_funcs:List[np.ufunc],prng_seed:int,passive_learning:bool=False, batch_size: int = 1, start_time:float=0)->np.ndarray: # type: ignore

    twenty_percent = int(len(X_train) * 0.20)
    fifty_percent = int(len(X_train) * 0.5)
    num_samples_axis = range(twenty_percent,fifty_percent+1,batch_size)
    # print(f"twenty_percent={twenty_percent}, fifty_percent={fifty_percent}")

    mse_vals: List[float] = [0.0]*len(num_samples_axis)

    prev_percent = None
    div_by_five_bool = False
    for idx, n_tr in enumerate(num_samples_axis):
        cur_percent = np.round(100*n_tr/len(X_train))
        # print(f"cur_percent={cur_percent}, n_tr={n_tr}")
        if cur_percent % 5 == 0 and cur_percent != prev_percent:
            prev_percent = cur_percent
            div_by_five_bool = True
            print(f'percentage complete: {cur_percent},\ttime elapsed since start={time.time() - start_time} seconds')
        elif batch_size != 1 and not div_by_five_bool:
            print(f'idx={idx}, n_tr={n_tr}, cur_percent={cur_percent}')

        theta_W = pool_alice(X_unlabeled=X_train,oracle=y_train,basis_funcs=basis_funcs,n_tr=n_tr,prng_seed=prng_seed,passive_learning=passive_learning)
        y_pred = predict(X_transformed=X_test_transformed,weights=theta_W)
        mse_vals[idx] = float(mean_squared_error(y_true=y_test, y_pred=y_pred))

    return np.array(mse_vals)

def get_correlations(batch: np.ndarray)->np.float_:

    results = np.array(stats.spearmanr(a=batch))
    spearmean_coefs = results[0,:,:]

    # make the diagonal np.nan so it will not contribute to the mean
    np.fill_diagonal(spearmean_coefs, np.nan, wrap=False)

    return np.nanmean(spearmean_coefs)