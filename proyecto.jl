using JuMP
using Ipopt
using SCS
using Juniper
using Mosek, MosekTools
using LinearAlgebra

print("\n\n\n\n\n\n\n\n\n\n")

nl_solver = optimizer_with_attributes(Ipopt.Optimizer, "tol" => 1e-6)
mip_solver = optimizer_with_attributes(SCS.Optimizer, "eps" => 1e-5)
juniper_optimizer = optimizer_with_attributes(Juniper.Optimizer, 
                                              "nl_solver" => nl_solver, 
                                              "mip_solver" => mip_solver)

B = 50e6
fc = 4e9
cl = 3e8
lambda = cl/fc
N = 3
Is = [1, 2, 3]
C = [1, 2]
Itdoa = [3, 5]

s0 = [0 0 0]

si = [2 1 2 ; 1 2 3 ; 2 3 4].*1e2
sc = [1 -5 2 ; 6 -5 4].*1e2
s = [[si] ; [sc]]

sigma0 = 10e-19

d = reduce(hcat,[[norm(s[1][i,:]-s[2][c,:]) for i in Is] for c in C])
d0 = [norm(s0'-s[2][c,:]) for c in C]

Ldb = -20*log10.(d).+(20*log10(lambda)+32.4)

h = ones(length(Is), length(C), N)

for n in 1:N
    h[:,:,n] = sqrt.(10 .^(Ldb./10))
end

Ac = zeros(length(C),length(Is), 3)

for c in C
    a = reduce(vcat,[(sc[c,:]'-si[i,:]')./d[i,c]-(sc[c,:]'-s0)./d0[c] for i in Is])
    for i in Is
        Ac[c,i,:] .= a[i,:]
    end
end

delta = ones(length(Is),length(C))

Pn = (4e-21)*B

# snir = zeros(length(Is),length(C))
Rcb = ones(length(Is),length(Is))*sigma0
gamma = ones(length(Is),length(C)).*1e-10
gamma0 = 5e-6;
gammam = 1e2;
Rcgen = zeros(length(C),length(Is),length(Is));

P = ones(length(Is)).*5e2

for i in Is
    # snir[i,:] .= ones(length(C)).*1e-10
    snirv = ones(length(C))
    grad = zeros(length(C))
    Rc = zeros(length(C),length(Is),length(Is))
    for c in C
        Rc[c,:,:].=Rcb+I(length(Is)).*(3 ./((gamma[:,c].^2).*(4*pi^2*B^2)))
        z = Ac[c,:,:]'*(Rc[c,:,:]^-1)
        grad[c] = -3*tr(z'*((Ac[c,:,:]'*(Rc[c,:,:]^-1)*Ac[c,:,:])^-2)z) ./(4*pi^2*B^2*gamma[i,c].^2)
    end

    while (sum(gamma[i,:] .< snirv .*gammam)>0)
        ci = argmin(grad .-(snirv.-1).*1e100)
        print("trabajando en: ",ci, "valido: ", snirv[ci], "\n")
        gamma[i,ci] = gamma[i,ci]+gamma0
        display(gamma[i,ci])

        # model = Model(SCS.Optimizer)
        # model = Model(juniper_optimizer)
        model = Model(Mosek.Optimizer)
        set_silent(model)

        @variable(model, Q[1:length(Is), 1:length(Is), 1:N])
        @variable(model, t >= 0)

        @objective(model, Max, t)

        # register(model, :tr, 1, tr; autodiff = true)


        H = zeros(length(Is), length(Is), N)
        for n in 1:N
            H[:,:,n] = h[:,:,n]*h[:,:,n]'
        end
        @constraint(model,[c1 = 1:length(C)], (1/gamma[i, c1])*(sum(sum(H[i1,c1,j1]*Q[i1,c1,j1] for j1 in 1:length(Is)) for i1 in 1:length(Is)))>= t*(sum(((1-(c1==c2))*(sum(sum(H[i1,c1,j1]*Q[i1,c1,j1] for j1 in 1:length(Is)) for i1 in 1:length(Is)))) for c2 in 1:length(C))+Pn))
        @constraint(model, [c1 = 1:length(C)],tr(Q[:,c1,:])==P[c1])
        for c1 in 1:length(C)
            @constraint(model, Q[:,c1,:] in PSDCone())
        end

        optimize!(model)

        tv = value.(t)
        display(tv)
        if tv >= 1
            Rc[ci,:,:].=Rcb+I(length(Is)).*(3 ./((gamma[:,ci].^2).*(4*pi^2*B^2)))
            z = Ac[ci,:,:]'*(Rc[ci,:,:]^-1)
            gd = (-3*tr(z'*((Ac[ci,:,:]'*(Rc[ci,:,:]^-1)*Ac[ci,:,:])^-2)z) ./(4*pi^2*B^2*gamma[:,ci].^2))
            # display(gd[ci])
            grad[ci] = gd[ci]
        else
            gamma[i,ci] = gamma[i,ci]-gamma0
            snirv[ci] = 0
        end
    end
end