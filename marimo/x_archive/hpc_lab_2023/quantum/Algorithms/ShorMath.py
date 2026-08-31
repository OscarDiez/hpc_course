import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # The Mathematics Behind Shor's Algorithm

        Shor's algorithm is an algorithm that can factorize big numbers quickly. For an integer $N$, Shor's algorithm can find a factor of it in $O((\log_{} N)^{3})$ which is an almost exponential speedup over classical algorithms. 

        This algorithm brought a lot of interest into building quantum computers, because if an ideal quantum computer could be built, then the widely used public-key cryptography RSA scheme could be broken.


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Classical part

        Shor's algorithm is hybrid becasuse it mixes a classical part and a quantum part. In this notebook, we will discuss the mathematics behind the classical part and assume the quantum part as a black box. 

        Firstly, we need to define what the algorithm does. It solves the factorization problem: given an intenger $N$, find two integers greater than one $P$ and $Q$ such that $PQ = N$, or state that $N$ is prime.

        Secondly, we need to define what the input of the algorithm is: it is $N$, an odd integer that is neither a prime nor the power of a prime. These assumptions are necessary for the algorithm to work, and when they are not respected factorizing is still easy.

        If $N$ is even, we can pick $P = 2$ and $Q = N/2$ so the even case is straightforward. 

        If $N$ is prime, we can check with primality tests such as [Miller-Rabin's](https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test) for primality in a reasonable time complexity so this case is also easier than solving general factorization.

        If $N$ is the power of a prime, we can check for every $2 \leq k \leq \log_{3}N$ if $N^{1/k}$ is an integer. If it is, then $P = N^{1/k}$ and $Q = N^{(k-1)/k}$ is a solution and the case for powers of primes is solved.

        Below are an implementation of a naive primality checking and the check for an exact power.
        """
    )
    return


@app.cell
def _():
    from math import gcd, log
    from random import randint

    def is_prime(N):
        """Returns if N is prime or not. Notice that this is not optimal,
        there is faster primality testing algorithms e.g. Miller-Rabin
        """
    
        if N == 2:
            return True  # only even prime
        if N % 2 == 0 or N <= 1:
            return False  # even numbers and 1 are not prime
    
        for i in range(3, N, 2):  # only try odd candidates
            if i*i > N:
                break  # we only need to check up to sqrt(N)
            if N % i == 0:
                return False  # found a factor
    
        return True

    def find_power_k(N):
        """Returns the smallest k > 1 such that N**(1/k) is an integer,
        or 1 if there is no such k.
        """
    
        upper_bound = int(log(N)/log(3))
    
        for k in range(2, upper_bound + 1):
            p = int(N**(1/k))
            if p**k == N:
                return k
    
        return 1
    return find_power_k, gcd, is_prime, randint


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Quantum part

        Shor's algorithm speedup comes from the quantum subroutine for period finding. To find the smallest $r$ for $a^{r} \equiv 1 \pmod N$, the algorithm uses Phase Estimation algorithm and the Quantum Fourier Transform to make calculations faster than possible with a classical computer. We will assume that subroutine as a blackbox for now.

        Despite that, we can implement a slower classical version that gives the same result as that blackbox. We test our code with $a = 2$ and $N=15$, which has a solution $r = 4$ because $2^{4} = 16$ and $16 \equiv 1 \pmod {15} $
        """
    )
    return


@app.function
def order_finding(a, N):
    """Returns the smallest r such that a^r = 1 mod N
    Notice that this is a naive classic implementation and is
    exponentially slower than the quantum version invented by Peter Shor.
    """
    
    i = 1
    a_i = a % N
    
    while a_i != 1:
        i += 1
        a_i = (a_i * a) % N
    
    return i


@app.cell
def _():
    order_finding(2, 15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Shor's algorithm

        Why focus on the smallest $r$ if it is easy to find an arbitrary one that works (e.g. $\phi(N)$)? The fact is that the smallest $r$ has a nice property that is key in our analysis. First, let's rewrite our equation:

        $$
        a^{r} - 1 \equiv 0 \pmod N \iff
        $$

        $$
        (a^{r/2} + 1)(a^{r/2} - 1) \equiv 0 \pmod N \iff
        $$

        $$
        N \mid (a^{r/2} + 1)(a^{r/2} - 1)
        $$

        Here, we assumed that $r$ is even, and that will be a requirement for our algorithm to work. Notice that $N \nmid (a^{r/2} - 1)$, because that would imply $a^{r/2} \equiv 1$ and violate our condition that $r$ was the smallest. Thus, the prime factors of $N$ are distributed among $a^{r/2} - 1$ and $a^{r/2} + 1$.

        There are two cases. In the first one, we're out of luck: if $N \mid (a^{r/2} + 1)$, then it might be that all the factors are on $a^{r/2} + 1$ and we can conclude nothing about the prime factorization of $N$.

        In the second case, $N \nmid (a^{r/2} + 1)$ and we can make a conclusion: because $N$ divides the product but not the numbers individually, at least one of $\gcd(N, a^{r/2} + 1)$ or $\gcd(N, a^{r/2} - 1)$ will be a non trivial factor of $N$. Our factorization is done.

        Thus, if finding $r$ is done quickly, we can try multiple values for $a$ until we found one that yields a factor. It can be shown that the probability that $a$ works is at least $1/2$, so with few attempts for $a$ we will find a factor. Shor's algorithm is concluded


        Given the math concepts behind Shor's algorithm, we can write the pseudo-code for the algorithm. Notice that in this step we are not worrying about the implementation of other parts of the algorithm: we assume that the classical parts of $\gcd$, primality-testing, checking if the k-th root is an integer are implemented and available for use. We also assume that the quantum subroutine for period finding is available for use. This yields the code:
        """
    )
    return


@app.cell
def _(find_power_k, gcd, is_prime, randint):
    def shor_algorithm(N):
        """Returns a pair of integers (P, Q) such that PQ = N for integer N"""
    
        if N % 2 == 0:  # even case
            return (N//2, 2)
    
        if is_prime(N):  # prime case
            return (N, 1)  # N is primes, factors cannot be found
    
        if find_power_k(N) > 1:  # prime power case
            P = int(N**(1/find_power_k(N)))  # we find a k such that N**(1/k) is an integer
            Q = N//P
            return (P, Q)
    
        # Now we can assume that the criteria for Shor's algorithm is met
    
        while True:
            # Non-deterministic, we will try multiple values for a
            a = randint(2, N-1)  # pick random a
        
            if gcd(a, N) != 1:  # Lucky case: a and N are not coprime!
                P = gcd(a, N)  # gcd yields a non-trivial factor
                Q = N//P
                return (P, Q)
        
            r = order_finding(a, N)  # quantum subroutine of the code
        
            if r % 2 == 0:
                continue

            P = gcd(a**(r//2) - 1, N)
            if P != 1:
                Q = N//P  # gcd yielded non trivial factor
                return (P, Q)
    return (shor_algorithm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Testing

        We can check if our code works! For example, let's take $N = {10013}$. That number has ${3}$ prime factors: ${17}$, ${19}$ and ${31}$. If we apply Shor's algorithm until we reach primality, we can find all of them.
        """
    )
    return


@app.cell
def _(shor_algorithm):
    N = 10013
    P, Q = shor_algorithm(N)
    print(
        "Shor's algorithm found {} = {} x {} which is {}!".format(N, P, Q, ["incorrect", "correct"][P*Q==N])
    )
    return (Q,)


@app.cell
def _(Q, shor_algorithm):
    S, T = shor_algorithm(Q)
    print(
        "Shor's algorithm found {} = {} x {} which is {} again!".format(Q, S, T, ["incorrect", "correct"][S*T==Q])
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

