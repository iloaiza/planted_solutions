import juliacall
from juliacall import Main as jl

jl.seval('import Pkg')
jl.seval('Pkg.add("QuantumMAMBO")')
jl.seval("using QuantumMAMBO")