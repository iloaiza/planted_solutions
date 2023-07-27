import QuantumMAMBO as QM
using SparseArrays, LinearAlgebra, Arpack

function get_gs(mat :: SparseMatrixCSC)
	if mat.m >= 4
		E_min, GS = eigs(mat, nev=1, which=:SR, maxiter = 500, tol=1e-4, ncv=minimum([50, mat.m]))
	else
		E,U = eigen(collect(mat))
		E = real.(E)
		E_min = minimum(E)
		GS = U[:,1]
	end

	return E_min, GS
end

function planted_cost_analysis(H, F_arr, do_eigs = true)
	Hmat = QM.to_matrix(H)
	num_F = length(F_arr)

	println("Collecting operators in two-body tensor...")
	println("H...")
	@time Hso = QM.F_OP_collect_obt(H)
	println("Fs...")
	@time Fso_arr = QM.F_OP_collect_obt.(QM.to_OP.(F_arr))

	Ltbt = zeros(num_F)
	vars = zeros(num_F)

	for i in 1:num_F
		Ltbt[i] = sum(abs2.(Hso.mbts[3] - Fso_arr[i].mbts[3]))
	end
	
	Lrelative = Ltbt ./ sum(abs2.(Hso.mbts[3])) 

	println("\n####################")
	println("Showing Ltbt and Lrelative...")
	@show Ltbt
	@show Lrelative
	println("####################\n")

	if do_eigs
		println("Doing full diagonalization...")
		@time E,U = eigen(collect(Hmat))
		GS = U[:,1]
		num_E = length(E)

		Lspectrum = zeros(num_F)
		for i in 1:num_F
			println("Diagonalizing F$i operator...")
			Fi = F_arr[i]
			Fmat = QM.to_matrix(Fi)
			HGS = Fmat * GS
			vars[i] = dot(HGS,HGS) - (dot(GS,HGS)^2)
			@time Ei,_ = eigen(collect(Fmat))
			Lspectrum[i] = sum(abs2.(E - Ei))
		end

		println("\n####################")
		println("Showing Lspectrum...")
		@show Lspectrum
		println("####################\n")

	else
		_,GS = get_gs(Hmat)
		for i in 1:num_F
			HGS = QM.to_matrix(F_arr[i]) * GS
			vars[i] = dot(HGS,HGS) - (dot(GS,HGS)^2)
		end
	end


	println("\n####################")
	println("vars...")
	@show vars
	println("####################\n")
end