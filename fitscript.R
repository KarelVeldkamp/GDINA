library(GDINA)


dta <- read.csv('~/Documents/GitHub/VAE_CDM/true/data/data_1.csv', header = F)
Z <- read.csv('~/Documents/GitHub/VAE_CDM/true/parameters/class_1.csv', header = F)
Q <- read.csv('~/Documents/GitHub/VAE_CDM/Qmatrix6d.csv', row.names = 1)
guess <- read.csv('~/Documents/GitHub/VAE_CDM/true/parameters/guess_1.csv', header = F)
slip <- read.csv('~/Documents/GitHub/VAE_CDM/true/parameters/slip_1.csv', header = F)

t1 = Sys.time()
dina = GDINA(dta, Q, model='DINA')
itempars = coef(dina, what='gs')
print(Sys.time()-t1)

mean(Z == personparm(dina))

plot(as.vector(guess), itempars[,1])

plot(as.vector(slip), itempars[,2])
