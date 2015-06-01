"""
The EM Algorithm
"""
import sys

class EMAlgorithm(object):
    """The expectation maximisation algorithm. Derivers are expected to
    fill in the expectation and maximisation steps"""
    def __init__(self):
        pass

    def compute_expectation(self, xs, params):
        """Compute the most likely values of the latent variables; returns lhood"""

        raise NotImplementedError

    def compute_maximisation(self, xs, Z, params):
        """Compute the most likely values of the parameters"""

        raise NotImplementedError

    def run(self, xs, params, iter_cb = None, iters=100, eps=1e-5):
        """Run with some initial values of parameters params"""

        lhood, Z = self.compute_expectation(xs, params)
        for i in xrange(iters):
            sys.stderr.write("Iteration %d, lhood = %f\n" % (i, lhood))
            params = self.compute_maximisation(xs, Z, params)
            # Add error and time to log
            if iter_cb is not None:
                iter_cb(i, params, lhood)

            lhood_, Z = self.compute_expectation(xs, params)
            if abs(lhood_ - lhood) < eps:
                sys.stderr.write("Converged with lhood=%f in %d steps.\n" % (lhood, i))
                lhood = lhood_
                break
            #elif lhood_ < lhood:
            #    raise RuntimeError("EM broken")
            else:
                lhood = lhood_

        return lhood, Z, params

