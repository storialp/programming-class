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

## Creating Numpy Arrays

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

## Indexing

When you get a slice of the array, what you get is called a view. This view is not a copy but the original data, modifying a view will modify the original array too!

```py
a = np.array([1, 2, 3])
b = a[0:1]
a.base is None # Evaluates to True since a is not a view of another array
b.base is a # Evaluates to True as b is a view of a
```

A lot of fucntions produce views too, like reshape or ravel. Know what you're using and how it works, when in doubt: check the docs!

If you need to get a new array that shares data but not references, use the .copy method

Not going into the most used methods for now, check the docs and practice, you'll memorize them anyway.

## Masking

You can select elements of an array that fulfill a given condition, this is called masking.

Interesting example:

```py
a = np.array([1, 2, 3, 4, 5])
mask = np.array([1, 0, 0, 1, 0], dtype=bool)
b = a[mask] # Will create a new array with the elements of a whose counterpart in mask evaluates to True meaning indices 0, 3
print(b) # Outputs np.array([1, 4]), remember this is not a view
c = a[[0, 3]] # Will also create a new array with the elements a[0] and a[3], this is called a list
```

Another interesting example:

```py
matrix = np.array([[1, 2], [3, 4]])
matrix[0] # Is a view!! Same as writing matrix[0, :] which is of course a slice
```

## Broadcasting

Most numpy methods are applied using what is called broadcasting. This means functions are applied element-wise. This is not the same behavior as in python but makes more sense when you view it from a linear algebra and matrix manipulation standpoint.

```py
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a + b # Evaluates to [5, 7, 9]
a * b # Evaluates to [4, 10, 18]
a * 3 # Evaluates to [3, 6,9]
```

You get the point. When you need to get the dot product of vectors or do matrix multiplication, there are specific functions for it.

Less intuitively:

Operators broadcast too

```py
a = np.array([1, 2, 3, 1])
b = np.array([1, 3, 4, 1])
a == b # Evaluates to np.array([True, False, False, True], dtype=bool)
# To check if two arrays have the same elements the method is:
np.array_equal(a,b) # Which of course will evaluate to False
```

If you try to perform an operation with two arrays of different shapes, it will work! It is quite unintuitive though as it expands the arrays into compatible shapes by repeating rows. Thankfully, this is seldom useful. IF you happen to need it, check docs and most importantly the slides for a concise explanation.

If you start working with complex numbers it is common to just get the complex number or a runtime warningwith nan, be careful and aware of what you are doing.

Never ever ignore warnings.
