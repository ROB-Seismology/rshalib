import numpy as np
import scipy.stats
import pylab
import hazard.rshalib as rshalib



M = 6.0
r = 10
PGAs = rshalib.utils.logrange(1E-3, 2.5, 1000)
log_PGAs = np.log10(PGAs)
PGA_test = 0.3

akb2010 = rshalib.gsim.AkkarBommer2010()
mean_pga = akb2010(M, r)
log_mean = np.log10(mean_pga)
log_sigma = akb2010.log_sigma(M, r)
P_exceedance = akb2010.get_exceedance_probability(PGA_test, M, r)[0]

dist = scipy.stats.truncnorm(-3, 3, log_mean, log_sigma)
pdf = dist.pdf(log_PGAs)
PGAs_exceedance = PGAs[PGAs > PGA_test]
PGAs_exceedance = np.concatenate([PGAs_exceedance[:1], PGAs_exceedance])
pdf_exceedance = pdf[PGAs > PGA_test]
pdf_exceedance = np.concatenate([[0], pdf_exceedance])
pylab.semilogx(PGAs, pdf)
pylab.fill(PGAs_exceedance, pdf_exceedance)
pylab.vlines(PGA_test, 0, pdf_exceedance[0], linestyles='--')
pylab.figtext(0.15, 0.8, "P(PGA > %s g) = %.6f" % (PGA_test, P_exceedance))
pylab.xlabel("PGA (g)")
pylab.ylabel("Probability density")
pylab.title("GMPE: %s, M=%.1f, d=%.0f" % (akb2010.name, M, r))
pylab.show()
