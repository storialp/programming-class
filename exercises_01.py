import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import random
import time

# %%

# First Exercise

hundred = np.int16(100)
u_hundred = np.uint16(100)

# Maximum of each type

max_int = ((2**16) / 2) - 1
min_int = -((2**16) / 2)
max_uint = 2**16 - 1
min_uint = 0

print(max_uint, min_uint, max_int, min_int)


cycling_signed = np.int16(max_int + 1)
print(cycling_signed, type(cycling_signed))

# %%

# Second exercise


def array_basics():
    py_list = [1, 2]
    np_arr = np.array(py_list)
    print(f"np_arr is of data type {np_arr.dtype}")
    py_l = [1.2, 3.5]
    np_a = np.array(py_l)
    print(f"np_a has data type {np_a.dtype}")
    print(np_arr)
    np_arr = np_arr.astype("float64")
    print(np_arr)
    print(f"np_a has data type {np_a.dtype}")
    mix_list = ["hello", 1, 2.5, True]
    mix_arr = np.array(mix_list)
    print(mix_arr, mix_arr.dtype)
    print(type(mix_arr[3]), isinstance(mix_arr[3], str))
    obj_list = [{1: "hello", 2: "world"}, (True, False), 2, "Yep"]
    obj_arr = np.array(obj_list)
    print(obj_arr, obj_arr[0])
    # To avoid type coersion
    unaltered_arr = np.array(mix_list, dtype=object)
    print(unaltered_arr, unaltered_arr.dtype)
    return


array_basics()


# %%

# Third exercise

def ones(n):
    array_of_ones = np.full(n, 1, dtype=float)
    return array_of_ones, type(array_of_ones), type(array_of_ones[0])


def small_int_ones(n):
    array_of_small_int_ones = np.full(n, 1, dtype="int64")
    return array_of_small_int_ones, type(array_of_small_int_ones), type(array_of_small_int_ones[0])


def big_int_ones(n):
    array_of_64int_ones = np.full(n, 1, dtype="int64")
    return array_of_64int_ones, type(array_of_64int_ones), type(array_of_64int_ones[0])


def bool_ones(n):
    array_of_bool_ones = np.full(n, 1, dtype=bool)
    return array_of_bool_ones, type(array_of_bool_ones), type(array_of_bool_ones[0])


def matrix_ones(n):
    matrix = np.full((n, n), 1)
    return matrix, type(matrix), type(matrix[0][1])


print(ones(5))
print(small_int_ones(5))
print(big_int_ones(5))
print(bool_ones(5))
print(matrix_ones(5))

# Could have used ones in every single one here too

# %%

# Fourth exercise


def full_minus(n):
    return np.full(n, -1)


def broadcast_minus(n):
    return np.ones(n) * -1


def conversion_minus(n):
    return np.array([-1 for i in range(n)])


# Missing one of the possible ways to do it but not important, just use ones
print(conversion_minus(5))

# %%

# Fifth exercise


def reading(l: list):
    n = len(l)
    arr = np.array(l)
    left, right = arr[: n // 2], arr[n//2:]
    print(f"First half is: \n {left} \n Second half is: \n {right}")
    # Trying to sum two arrays of different sizes will throw exception
    concat = np.concatenate((left, right))
    hstack = np.hstack((left, right))
    print(concat, hstack)
    print(concat.base, hstack.base)  # They both return new arrays not views
    # As with all operators, comparison is broadcasted not very useful rn
    print(concat == arr, concat == hstack)
    left[1] = 30
    # Of course it changes the original array, expected behaviour
    print(arr, left.base)
    return


reading([1, 2, 3, 4, 5])


# %%

# Sixth exercise

def strides_slicing(l: list):
    arr = np.array(l)
    even_indices = arr[::2]
    odd_indices = arr[1::2]
    # Of course this produces views, if we wanted new arrays, use .copy method
    print(even_indices, odd_indices)
    return


strides_slicing([0, 1, 2, 3, 4, 5, 6])


# %%

# Seventh exercise

def strides_sequel(l: list):
    arr = np.array(l)
    n = len(arr)
    reverse_first = arr[(n - 1)//2::-1]
    reverse_second = arr[n:(n - 1)//2:-1]
    print(reverse_first, reverse_second)
    reverse_even = arr[(n-2)+(n % 2)::-2]
    reverse_odd = arr[(n-1)-(n % 2)::-2]
    print(reverse_even, reverse_odd)
    return


strides_sequel([0, 1, 2, 3, 4, 5, 6, 7])


# %%

# Eigth exercise

def indexing(l: list):
    arr = np.array(l)
    n = len(arr)
    indices = [0, n//2, n-1]
    selected = arr[indices]
    print(arr[indices], arr[indices].base)
    return


# Of course, as we saw before this creates a new array
indexing([0, 1, 2, 3, 4, 5, 6])

# %%

# Ninth exercise


def indexing_two(l: list, m: int):
    arr = np.array(l)
    n = len(arr)
    rand_indices = np.random.randint(0, n, size=m)
    rand_arr = arr[rand_indices]
    print(rand_arr)
    return


# Reminds me of bootstrapping
indexing_two([1, 24, 564, 67, -2, -5, 5], 12)
indexing_two([1, 24, 564, 67, -2, -5, 5], 4)
indexing_two([1, 24, 564, 67, -2, -5, 5], 7)

# %%

# Tenth exercise


def mask_indexing(l: list):
    arr = np.array(l)
    n = len(arr)
    new_arr = arr[arr > 0]
    print(new_arr, new_arr.base)
    return


mask_indexing([-1, 2, -3, 7, 6, -1, 0, 45])

# %%

# Eleventh exercise


def mask_indexing_two(l: list, r: float):
    arr = np.array(l)
    n = len(arr)
    s = np.random.random(n)
    indices = s < r
    indexed = arr[indices]
    trues = indices.sum()
    print(indexed, trues)
    return indexed


def repeats(l: list, r: float, iterations: int = 100):
    length_sum = 0
    for i in range(iterations):
        a = mask_indexing_two(l, r)
        length_sum += len(a)
    average = length_sum / iterations
    expectation = len(l) * r
    print(average, expectation)
    return


mask_indexing_two([-1, 2, -3, 7, 6, -1, 0, 45], 0.678)
# Indeed for a very large number of iterations, it gets closer and closer to expectation
repeats([-1, 2, -3, 7, 6, -1, 0, 45], 0.678, 1000000)

# %%

# Twelfth exercise

floats_arr = np.random.random(5)
print(floats_arr)
floats_arr[3] = 20
print(floats_arr)
# Of course 20 gets coerced into floating point value
floats_arr[4] = 1.8e308
print(floats_arr, floats_arr.dtype)
# for the default float64 dtype, 1.8e308 is the sentinel value
int_arr = np.random.randint(0, 10, size=5)
print(int_arr, int_arr.dtype)
int_arr[1] = 2.6
print(int_arr, int_arr.dtype)
# 2.6 is rounded down and coerced into int
# Now trying special values
int_arr[2] = np.inf
print(int_arr, int_arr.dtype)
int_arr[2] = np.nan
print(int_arr, int_arr.dtype)
# They both throw errors

# %%

# Thirteenth exercise


def assigning(l: list):
    arr = np.array(l)
    n = len(arr)
    right = arr[n-1:n//2 - 1:-1]
    left = arr[(n//2)-1::-1]
    new = np.concatenate((left, right))
    print(new, new.base)
    return


assigning([-1, 2, -3, 7, 6, -1, 0, 45])

# %%

# Fourteenth exercise


def list_assigning(l: list):
    arr = np.array(l)
    n = len(arr)
    arr[[0, -1, n//2]] = [0, 2, 1]
    print(arr)
    return


list_assigning([-1, 2, -3, 7, 6, -1, 0, 45])
list_assigning([-1])

# %%

# Fifteenth exercise


def mask_assigning(l: list):
    arr = np.array(l)
    arr[arr < 0] = arr[arr < 0] * -1
    print(arr)
    return


mask_assigning([-1, 2, -3, 7, 6, -1, 0, -45])

# %%

# Sixteenth exercise


def broadcast_assignment(n: int):
    arr = np.ones(n)
    arr[::2] = arr[::2] * 0
    print(arr)
    return


broadcast_assignment(7)

# %%

# Seventeenth exercise


def broadcasted_assignment_two(n: int):
    arr = np.zeros(n)
    arr[:n//2] = arr[:n//2] + 1
    print(arr)
    return


broadcasted_assignment_two(7)

# %%

# Eighteenth exercise


def miscellaneous(n):
    not_tau_arr = np.linspace(0.0, 2*np.pi, num=n, endpoint=False)
    # Alternatively while excluding 2pi use arange
    arr = np.arange(0.0, 2*np.pi, step=2*np.pi / n)
    # Including 2pi
    tau_arr = np.linspace(0.0, 2*np.pi, num=n, endpoint=True)
    sines = np.sin(tau_arr)
    cosines = np.cos(tau_arr)
    print(sines)
    print(cosines)
    print(arr)
    plt.plot(arr, sines, label="Sine")
    plt.plot(arr, cosines, label="Cosine")
    plt.legend()
    plt.xlabel("Radians")
    return


miscellaneous(100)


# %%

# Nineteenth exercise

def normal_random(n: int):
    t0 = time.time()
    py_rnd = [random.gauss(0.0, 1.0) for i in range(n)]
    t1 = time.time()
    np_rnd = np.random.normal(0.0, 1.0, n)
    t2 = time.time()
    print(
        f"Milliseconds taken by python to generate {n} random numbers {t1-t0}")
    print(
        f"Milliseconds taken by numpy to generate {n} random numbers {t2-t1}")
    return


normal_random(1000000)


# %%

# Twentieth exercise

def mix(n):
    two_d_rand = np.random.normal(0.0, 1.0, (10, 20))
    even_pairs = two_d_rand[::2, ::2].copy()
    ev_pairs_mu = (even_pairs.sum() / even_pairs.size)
    # Doing this process for n iterations
    full_result = np.zeros(n)
    sub_result = np.zeros(n)
    for i in range(n):
        random_arr = np.random.normal(0.0, 1.0, (10, 20))
        random_sub_arr = random_arr[::2, ::2]
        full_mu = random_arr.sum() / random_arr.size
        sub_mu = random_sub_arr.sum() / random_sub_arr.size
        full_result[i] = full_mu
        sub_result[i] = sub_mu
    plt.hist(full_result)
    plt.show()
    plt.clf()
    plt.hist(sub_result)
    plt.show()
    plt.clf()
    plt.plot(full_result)
    return


mix(100000)

# %%

# Twenty-first exercise


def stacking(n):
    l = [np.arange(i, i+5) for i in range(n)]
    arr = np.array(l)
    flat = arr.flatten()
    print(l, arr, flat)
    return


stacking(5)

# %%

# Twenty-third exercise


def advanced_broadcasting(n):
    result = np.zeros((n, 5))
    broadcaster = np.arange(0, 5)
    # Ideally done without a loop but can not figure it out rn
    for i in range(n):
        result[i] = result[i] + (broadcaster + i)
    print(result)
    return


advanced_broadcasting(5)


# %%

# Twenty-fourth exercise

def sieve_of_eratosthenes(n):
    bools = np.ones(n, dtype=bool)
    bools[0], bools[1] = False, False
    for i in range(2, int(np.sqrt(n-1)) + 1):
      # TODO: finish
        pass
    print(bools)
    return


sieve_of_eratosthenes(10)


# %%

# Twenty-sixth exercise
