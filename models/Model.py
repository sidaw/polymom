"""
Base class for generative models
"""
import scipy as sc
import time
import tempfile
import os, shutil

class Model:
    """Generic mixture model that contains a bunch of weighted means"""
    def __init__(self, **params):
        """Create a mixture model for components using given weights"""
        self.params = params
        # Directory for storing memmapped data
        self.dname = None

    def __del__(self):
        """Annihilate the temporary directory"""
        if self.dname is not None:
            shutil.rmtree(self.dname)
            self.dname = None

    def set_seed(self, seed = None):
        """Set seed or generate a new one"""
        if seed is None:
            seed = int(time.time() * 1000)
        sc.random.seed(int(seed))
        self["seed"] = seed

    # Container interface
    def __getitem__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            raise AttributeError()

    def __setitem__(self, name, value):
        self.params[name] = value

    def __delitem__(self, name, value):
        raise RuntimeError("Can't delete parameters")

    def __len__(self):
        return len(self.params)

    def _allocate_samples(self, name, shape):
        """Allocate for (shape) samples"""
        # Save samples in a temporary mem-mapped array, fname save in
        # the metadata "params"

        if self.dname is None:
            self.dname = tempfile.mkdtemp()
        arr = sc.memmap(os.path.join(self.dname,name), mode="w+", shape=shape, dtype = sc.double)
        return arr

    def save(self, fname):
        """Flush to disk"""
        sc.savez(fname, seed = self["seed"], **self.params)

    @staticmethod
    def from_file(fname):
        """Load model from a npz file"""

        params = dict(sc.load(fname).items())
        model = Model(fname, **params)
        if "seed" in params:
            model.set_seed(model["seed"])
        return model

    def exact_moments(self, terms):
        """
        Compute exact moments for terms
        """
        raise NotImplementedError()

    def empirical_moments(self, xs, terms):
        """
        Estimate moments from data xs for terms
        """
        raise NotImplementedError()

    def sample(self, n_samples):
        """
        return n_samples of data
        """
        raise NotImplementedError()

    def llikelihood(self, xs):
        """
        return the log-likelihood of data
        """
        raise NotImplementedError()

    def exact_moment_equations(self, maxdeg):
        """
        return scipy moment equation expressions
        """
        raise NotImplementedError()

    def empirical_moment_equations(self, xs, maxdeg):
        """
        return scipy moment equation expressions
        """
        raise NotImplementedError()

    def moment_monomials(self, maxdeg):
        """
        return scipy moment monomials
        """
        raise NotImplementedError()

