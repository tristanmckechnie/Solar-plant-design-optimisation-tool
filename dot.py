# Wrapper to call DOT from Python
#
# Will work on both Linux and Windows platforms.  There 
# should not be any need to change this file.
# ---------------------------------------------------------
import platform
import numpy as nm
import ctypes as ct
from ctypes import byref as B

# Define the Python wrapper to the DOT shared library as a Python class
class dot:
      
    #Set some local constants
    nInfo   = 0
    nMethod = 1
    nPrint  = 1
    nMinMax = 0
    nMaxInt = 20000000
    nmParam = nm.empty(1, float)
    nmRPRM  = nm.zeros(20, float)
    nmRPRM[8] = 0.01
    nmRPRM[9] = 0.001
    nmIPRM  = nm.zeros(20, int)
    
    nmIDISCR = nm.zeros(2*1000, int)
    nmDISCRT = 0

    # ---------------------------------------------------------
    # Initialize the class - this loads the DOT shared library
    # according to the platform detected at run time
    def __init__(self,nDvar):
        
        self.nDvar = nDvar
        
        # arrays for electric, dump and storage
        self.electric_heat = nm.zeros(nDvar)
        self.dumped_heat = nm.zeros(nDvar)
        self.thermal_storage = nm.zeros(nDvar)
        
        self.systemName = platform.system()
        
        # Load the shared library on Linux
        if ( self.systemName == 'Linux' ):
            self.dotlib = ct.cdll.LoadLibrary("libDOT2.so")
            
        # Load the shared library on Windows
        elif ( self.systemName == 'Windows' ):
            self.dotlib = ct.windll.LoadLibrary("DOT.dll")
            
        # Else throw an exception to indicate that no supported
        # platform was found
        else :
            raise ValueError( 'Unsupported Operating System: '+self.systemName )

    # ---------------------------------------------------------
    # The DOT wrapper itself - called by the user to start the
    # DOT optimization
    # Returns an array that contains the objective function, worst constraint, all the design
    # variable values
    def dotcall(self, x, xl, xu, nCons):

        # Reset nInit
        nInit = 0

        #Initailize all array types
        nDvar  = x.shape[0]
        ctDVAR = ct.c_double * nDvar
        ctCONS = ct.c_double * nCons
        ctRPRM = ct.c_double * 20
        ctIPRM = ct.c_int * 20
        
        ctIDISCR = ct.c_double * 2000 # bigdot stuff
        

        #Initialize all arrays
        RPRM = ctRPRM(*(self.nmRPRM))   #Tells dot to use defaults
        IPRM = ctIPRM(*(self.nmIPRM))   #Tells dot to use defaults
        X    = ctDVAR(*(x))             #Initial values
        XL   = ctDVAR(*(xl))            #Lower bounds
        XU   = ctDVAR(*(xu))            #Upper bounds
        G    = ctCONS(*([0.0]*nCons))   #Constraints
        
        # BIGDOT arrays

        IDISCR = ctIDISCR(*([0.0]*2000)) # bigdot stuff


        #Initialize constants
        METHOD  = ct.c_int( self.nMethod )
        NDV     = ct.c_int( nDvar )
        NCON    = ct.c_int( nCons )
        IPRINT  = ct.c_int( self.nPrint )
        MINMAX  = ct.c_int( self.nMinMax )
        INFO    = ct.c_int( self.nInfo )
        OBJ     = ct.c_double( 0.0 )
        MAXINT  = ct.c_int( self.nMaxInt )
        
        DISCRT = ct.c_int(self.nmDISCRT) # bigdot stuff
        
        # Call DOT510 to determine memory requirements for DOT and allocate the memory
        # in the work arrays
        NRWK    = ct.c_int()
        NRWKMN  = ct.c_int()
        NRIWD   = ct.c_int()
        NRWKMX  = ct.c_int()
        NRIWK   = ct.c_int()
        NSTORE  = ct.c_int()
        NGMAX   = ct.c_int()
        IERR    = ct.c_int()
        

        if ( self.systemName == 'Linux' ):
            self.dotlib.dot510_(B(NDV), B(NCON), B(METHOD), B(NRWK), B(NRWKMN), B(NRIWD), B(NRWKMX), B(NRIWK), B(NSTORE), B(NGMAX), B(XL), B(XU), B(MAXINT), B(IERR))    
        elif ( self.systemName == 'Windows' ):
            self.dotlib.DOT510(B(NDV), B(NCON), B(METHOD), B(NRWK), B(NRWKMN), B(NRIWD), B(NRWKMX), B(NRIWK), B(NSTORE), B(NGMAX), B(XL), B(XU), B(MAXINT), B(IERR))
        else :
            raise ValueError( 'Unsupported Operating System: '+self.systemName )

        ctRWK  = ct.c_double * NRWKMX.value
        ctIWK  = ct.c_int * NRIWK.value
        IWK    = ctIWK( *([0]*NRIWK.value) )
        WK     = ctRWK( *([0.0]*NRWKMX.value) )

        # Call DOT itself
        while (True):
            if ( self.systemName == 'Linux' ):
                self.dotlib.dot_(B(INFO),B(METHOD),B(IPRINT), B(NDV),  B(NCON), B(X), B(XL), B(XU), B(OBJ), B(MINMAX), B(G), B(RPRM), B(IPRM), B(WK), B(NRWKMX), B(IWK), B(NRIWK))
            elif ( self.systemName == 'Windows' ):
                self.dotlib.DOT(B(INFO),B(METHOD),B(IPRINT), B(NDV),  B(NCON), B(X), B(XL), B(XU), B(OBJ), B(MINMAX), B(G), B(RPRM), B(IPRM), B(WK), B(NRWKMX), B(IWK), B(NRIWK)) # bigdot: ALLDOT(...,B(DISCRT), B(IDISCR))                      
            else :
                raise ValueError( 'Unsupported Operating System: '+self.systemName )
            if ( INFO.value == 0 ) :
                break
            else:
                self.evaluate(X, OBJ, G, self.nmParam) # self.electric_heat, self.dumped_heat, self.thermal_storage =

        # Process the DOT output into a return value array
        rslt = nm.empty( 2+nDvar, float)
        rslt[0] = OBJ.value
        rslt[1] = 0.0
        if len(G) > 0 :
            rslt[1] = max(G)
        for i in range( nDvar ):
            rslt[2+i] = X[i]
        return rslt #, self.electric_heat, self.dumped_heat, self.thermal_storage

    # ---------------------------------------------------------
    # Function to evaluate the objective function and constraints - this is just a test function
    # and should be provided by the user
    def evaluate(self, x, obj, g, param):
        
        return
