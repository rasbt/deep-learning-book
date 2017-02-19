""" Python Tools
"""
# Sebastian Raschka 2016-2017
#
# ann is a supporting package for the book
# "Introduction to Artificial Neural Networks and Deep Learning:
#  A Practical Guide with Applications in Python"
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import time
import sys


def progress(iteritem, update=1, stderr=False, start_newline=True):
    """Iteration wrapper for printing progress, time left, and time remaining.

    Parameters
    ----------
    iteritem : iterable
        An item iterated over in for loop.
    update : int (default: 1)
        Update interval in seconds.
    stderr : Bool (default: False)
        Output stream. Uses `sys.stdout` if `False' (default) or `sys.stderr`
        if `True`.
    start_newline : bool (default: True)
        Prints progress on a new line if `True` (default)

    Yields
    -------
    generator
        A generator object containing the current item from `iteritem`

    Examples
    --------
    >>># for i in progress(range(5000000)):
    #...    pass
    >>>
    """
    if stderr:
        stream = sys.stderr
    else:
        stream = sys.stdout
    start_time = time.time()
    curr_iter = 0
    if start_newline:
        stream.write('\n')

    max_iter = len(iteritem)
    dlen = len(str(max_iter))
    memory = 0
    for idx, item in enumerate(iteritem):

        elapsed = int(time.time() - start_time)

        curr_iter += 1
        not_update = elapsed % update

        if not not_update and elapsed != memory:
            memory = elapsed
            remain = (max_iter - curr_iter) * (curr_iter / elapsed)
            out = '\r%*d/%*d | Elapsed: %d sec | Remaining: %d sec           '\
                % (dlen, curr_iter, dlen, max_iter, elapsed, remain)
            stream.write(out)
            stream.flush()

        yield item

    out = '\r%*d/%*d | Elapsed: %d sec | Remaining: 0 sec           '\
        % (dlen, curr_iter, dlen, max_iter, elapsed)
    stream.write(out)
    stream.flush()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
