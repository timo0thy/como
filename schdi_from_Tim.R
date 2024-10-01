
library(fitdistrplus)
library(copula)
library(readxl)
library(copula)
library(scatterplot3d)
library(ggplot2)
library(copBasic)
library(grid) 
library(TSA)
library(forecast)
library('ggplot2')
library('VineCopula')
library('R.matlab')
library('VineCopula')
library('climod')

# read data
Copula_input <- read_excel("/Users/tianyl/Desktop/Study/4_summer school in como/sensitivity test/Copula_input.xlsx")
Copula_input <- as.data.frame(Copula_input)
JJA_t2m  <- Copula_input[,3]
JJA_wsd  <- Copula_input[,4]

#####################################


# scale data to 0~1 by calculating pnorm
tmp <-JJA_t2m  
wsd <-JJA_wsd 

x=c(tmp)
y=c(wsd)


Ft1 <- fitdist(x, "norm")
gofstat(Ft1)
u <- pnorm(x,mean = Ft1$estimate[1], sd = Ft1$estimate[2])
plot(x,u)

Fw1 <- fitdist(y, "norm")
gofstat(Fw1 )
v <- pnorm(y,mean = Fw1$estimate[1], sd = Fw1$estimate[2])
plot(y,v)

########################   ###########################


# selectedCopula
selectedCopula <- BiCopSelect(u, v, familyset = NA, selectioncrit = "AIC", indeptest = FALSE, level = 0.05, weights = NA, rotations = TRUE, se = FALSE, presel = TRUE, method = "mle" )

# print selectedCopula
selectedCopula
selectedCopula$AIC
selectedCopula$familyname

# GOF test with selected copula
BiCopGofTest( u, v, family=1, par = -0.68, method = "kendall", B = 100, obj = NULL )

# calculate SCDHI

for (i in 1:798){
  P_t[i] <- pnorm(x[i], mean = Ft1$estimate[1], sd = Ft1$estimate[2])
  P_w[i] <- pnorm(y[i], mean = Fw1$estimate[1], sd = Fp1$estimate[2])
}
P_tw <- P_w-BiCopCDF(P_t, P_w, selectedCopula) #P(WSD<w,T>t)

SCDHI <- qnorm(P_tw)
scatterplot3d(x,y,SCDHI)
plot(JJA_t2m,SCDHI)
plot(JJA_wsd,SCDHI)

write.csv(SCDHI, file = "SCDHI.csv")

# calculate SCDHI with kendall distribution

for (i in 1:798){
  P_t <- pnorm(x[i], mean = Ft1$estimate[1], sd = Ft1$estimate[2])
  P_w <- pnorm(y[i], mean = Fp1$estimate[1], sd = Fp1$estimate[2])
}

P_tw_NEW <- P_w-kfuncCOP(BiCopCDF(P_t, P_w, selectedCopula) ,cop=P)

plot(P_tp ,P_tp_NEW)

SCDHI_NEW <- qnorm(P_tp_NEW)
scatterplot3d(x,y,SCDHI_NEW)
plot(SCDHI ,SCDHI_NEW)
write.csv(SCDHI, file = "SCDHI_KC.csv")

