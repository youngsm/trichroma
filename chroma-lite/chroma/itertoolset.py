from itertools import *
import collections
from copy import deepcopy

def peek(iterable):
    """Peek at the first element of `iterable`.

    Returns a tuple of the form (first_element, iterable).

    Once peek() has been called, the original iterable will be modified if it
    was an iterator (it will be advanced by 1); use the returned iterator for
    an equivalent copy of the original iterable.
    """
    it = iter(iterable)
    first_element = next(it)
    return first_element, chain([first_element], it)

def repeatfunc(func, times=None, *args):
    """Repeat calls to func with specified arguments.

    Example: repeatfunc(random.random)
    """
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))

def repeatcopy(object, times=None):
    """Returns deep copies of `object` over and over again. Runs indefinitely
    unless the `times` argument is specified."""
    if times is None:
        while True:
            yield deepcopy(object)
    else:
        for i in range(times):
            yield object

def repeating_iterator(i, nreps):
    """Returns an iterator that emits each element of `i` multiple times
    specified by `nreps`. The length of this iterator is the lenght of `i`
    times `nreps`.  This iterator is safe even if the item consumer modifies
    the items.

    Examples:
        >>> list(repeating_iterator('ABCD', 3)
        ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D']
        >>> list(repeating_iterator('ABCD', 1)
        ['A', 'B', 'C', 'D']
    """
    for item in i:
        for counter in range(nreps):
            yield deepcopy(item)

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def roundrobin(*iterables):
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return chain.from_iterable(repeat(tuple(iterable), n))

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def consume(iterator, n=None):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)
