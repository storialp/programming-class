import time

def fib(n, memo={}):
    print(f"time at fib start{time.now()}")
    if n <= 1:
        return 1
    elif n in memo:
        return memo[n]
    else:
        memo[n] = fib(n-1) + fib(n-2)
        return memo[n]
print(fib(35))

def naive_fib(n):
    if n <= 1:
        return 1
    else:
        return naive_fib(n-1) + naive_fib(n-2)
    
print(naive_fib(35))


#%%

def grid_traveler(m, n, memo= {}):
    if n == 0 or m == 0:
        return 0
    elif n == 1 and m == 1:
       return 1
    elif (m, n) in memo:
        return memo[(m, n)]
    else:
        memo[(m, n)] = grid_traveler(m, n -1) + grid_traveler(m-1, n)
        return memo[(m, n)]
   
print(grid_traveler(100,100))

#%%

# Knapsack pattern-ish i think

def can_sum(n:int, arr: list):
    in n == 0:
        return True
    for i in range(len(arr)):
        can_sum(n - arr[i])
    
