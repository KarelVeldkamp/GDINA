library(GDINA)

generate_Q_row <- function() {
  num_ones <- sample(1:3, 1)
  row <- c(rep(0, 10 - num_ones), rep(1, num_ones))
  sample(row)
}

# Creating a binary matrix with 100 rows and 10 columns
Q <- data.frame(matrix(NA, nrow = 100, ncol = 10))

# Filling the matrix with random rows
for (i in 1:100) {
  Q[i,] <- generate_Q_row()
}

gs <- data.frame(guess=rbeta(15, 10,40),slip=rbeta(15, 10, 40))

sim = simGDINA(1000, Q, model='DINA', gs.parm = gs)

data = extract(sim, what='dat')
prob_true = extract(sim,what = "catprob.parm")
att_true = extract(sim,what = "attribute")


fit = GDINA(data, Q, model='DINA')
est = coef(fit, 'gs')
mean(personparm(fit) == att_true)



mse = function(a,b){
  mean((a-b)^2)
}
mseg = mse(gs$guess, est[,1])
mses = mse(gs$slip, est[,2])
par(mfrow=c(1,2))
plot(gs$guess, est[,1], main=paste('guess: ', mseg))
abline(0,1)
plot(gs$slip, est[,2], main=paste('slip: ', mses))
abline(0,1)
par(mfrow=c(1,1))



Q = matrix(c(0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,0,1,0,0,0,1,1,0,0,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,1,1,1), byrow = T, ncol=4)
write.csv(Q, file='~/Documents/GitHub/VAE_CDM/Qmatrix.csv')


Q = matrix(c(rep(c(1,1), 5), rep(c(0,1), 5), rep(c(1,0),5)), ncol=2, byrow=T)
write.csv(Q, file='~/Documents/GitHub/VAE_CDM/Qmatrix2d.csv')

library(gdina)
?dina
