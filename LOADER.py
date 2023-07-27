#WORKFLOW FOR LOADING SAVED PLANTED SOLUTIONS, GENERATED USING JULIA CODE MF_planted.jl
import juliacall
from juliacall import Main as jl

jl.seval("using QuantumMAMBO")

mambo = jl.QuantumMAMBO
import openfermion as of
from openfermion import FermionOperator

DATAFOLDER = "./SAVED/"

mol_name = "lih"
method = "DF-boost"

fname = DATAFOLDER + mol_name + ".h5"

frag = QM.load_frag(fname, method)
H_planted = QM.to_OF(frag) #openfermion planted Hamiltonian object