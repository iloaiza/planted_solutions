import QuantumMAMBO as QM

MOL_LIST = ["h2", "h4", "lih", "beh2", "h2o", "nh3", "n2"]

include("analysis.jl")

do_one_body = false #whether will also generated planted solutions for collected one-body tensor, very inefficient!
spectral_analysis = false #whether spectral analysis is done, extremely expensive (full Fock space diagonalizations)

for mol in MOL_LIST
	println("Starting workflow for $mol")
	H,_ = QM.SAVELOAD_HAM(mol,"",false)
	H.mbts[1] .= 0
	H.filled[1] = false

	fname = "./SAVED/$mol.h5"

	println("Obtaining planted solutions...")
	println("DF-boosted:")
	@time F_DFB = QM.MF_planted(H, method="DF-boost", OB = false)
	QM.save_frag(F_DFB, fname, "DF-boost")

	println("DF:")
	@time F_DF = QM.MF_planted(H, method="DF", OB=false)
	QM.save_frag(F_DF, fname, "DF")

	println("GFR:")
	@time F_GFR = QM.MF_planted(H, method="CSA")
	QM.save_frag(F_GFR, fname, "GFR")

	F_arr = [F_DFB, F_DF, F_GFR]

	if include_OB
		println("Obtaining planted solutions for OB=true")
		println("DF-boosted:")
		@time F2_DFB = QM.MF_planted(H, method="DF-boost", OB=true)
		QM.save_frag(F2_DFB, fname, "OB_DF-boost")

		println("DF:")
		@time F2_DF = QM.MF_planted(H, method="DF", OB=true)
		QM.save_frag(F2_DF, fname, "OB_DF")

		push!(F_arr, F2_DFB)
		push!(F_arr, F2_DF)
		println("Array order is DFB, DF, GFR, DFB+OB, DDF+OB")
	else
		println("Array order is DFB, DF, GFR")
	end

	planted_cost_analysis(H, F_arr, spectral_analysis) #set to true to also do spectral analysis, requires full diagonalization on Fock space!
end


