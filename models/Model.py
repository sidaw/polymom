"""
The parameters are stored as is in an npz file, and the data is stored
in a memmappable npy file, as objects, while the data is compressed.
"""
import scipy as sc
import time
import tempfile
import os, shutil

class Model:
    """Generic mixture model that contains a bunch of weighted means"""

    def __init__( self, fname, **params ):
        """Create a mixture model for components using given weights"""
        self.fname = fname
        self.params = params
        # Directory for storing memmapped 
        self.dname = None

    def __del__( self ):
        """Annihilate the temporary directory"""
        if self.dname is not None:
            shutil.rmtree(self.dname)
            self.dname = None

    def set_seed( self, seed = None ):
        """Set seed or generate a new one"""
        if seed is None:
            seed = int(time.time() * 1000)
        sc.random.seed( int( seed ) )
        self.add_parameter( "seed", seed )

    def add_parameter( self, name, values ):
        """Add a parameter with values as a whole object. No compression"""
        self.params[ name ] = values

    def get_parameter( self, name ):
        """Read the parameter value from the store"""
        v = self.params[ name ]
        return v

    def _allocate_samples( self, name, shape ):
        """Allocate for (shape) samples"""
        # Save samples in a temporary mem-mapped array, fname save in
        # the metadata "params"

        if self.dname is None:
            self.dname = tempfile.mkdtemp()
        arr = sc.memmap( os.path.join(self.dname,name), mode="w+", shape=shape, dtype = sc.double )
        return arr

    def save(self):
        """Flush to disk"""
        sc.savez( self.fname, **self.params )

    @staticmethod
    def from_file( fname ):
        """Load model from a HDF file"""

        if not fname.endswith(".npz"):
            fname += ".npz"
        params = dict( sc.load( fname ).items() )
        model = Model( fname, **params )
        if "seed" in params:
            model.set_seed( model.get_parameter("seed") )
        return model

