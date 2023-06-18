Intro to Numpy and Scipy

// Treating this kind of as a notebook because i dont really need to run code for now

Numpy: library that extends python and makes it easier to do computations, especially with matrices and vectors (1d and 2d arrays respectively)

It is faster because basically all its methods are implemented in C

Integers and floats in Numpy are of a specified size and type

```py
import numpy as np
np_integer = np.int64(32)
np_float = np.float64(2.5)
py_integer = 32

type(np_integer) # numpy.int
type(py_integer) # int

py_integer == np_integer # Evaluates to True anyway

```

Big difference you **have** to keep in mind is that because numbers are of specified size, you can and will overflow if you are not careful.

Remember the magic 2^bits formula for possible combinations and whether the number is signed or not

# Creating Numpy Arrays

```py
python_list = [2, 3, 7]
np_array = np.array(python_list)

type(np_array) # Evaluates to numpy.ndarray
type(np_array[0]) # Evaluates to numpy.int64

py_2dlist = [[1,2], [3,4]]
np_array = np.array(py_2dlist)

a[0, 1] # Evaluates to 2
a.shape  # Evaluates to (2,2)
a.size # Number of elements in matrix, evaluates to 4
```

All elements in an np array are of the same type (oversimplification but mostly true)

Why is this so much faster?

Any time you perform an operation in python, the interpreter has to check the type of the variable, look for the appropriate method based on the type and more things of the sort.

Adding up two integers and two floats are not the same instructions in machine code, python has to take this into account

If you know the type of every element, a lot of work can be skipped. Besides, numpy arrays are true arrays, not lists. Operating on contiguous chukns of memory is much faster.

Not only can a lot of work be skipped, it can also be parallelized. This is called vectorization.

// C side note: they also don't require dereferencing operations, still don't fully understand this. Def: accessing a value referred to by a pointer. Seems to be especially useful when allocating memory dynamically (i.e. malloc) in C. This is because when you allocate memory dynammically in C you get a pointer to the beginning of the allocated block.

# Indexing

When you get a slice of the array, what you get is called a view. This view is not a copy but the original data, modifying a view will modify the original array too!

A lot of fucntions produce views too, like reshape or ravel. Know what you're using and how it works, when in doubt: check the docs!

If you need to get a new array that shares data but not references, use the .copy method
