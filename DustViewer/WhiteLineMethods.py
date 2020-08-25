# Created 2014, Zack Gainsforth
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import io
from PIL import Image
from QuickPlot import QuickPlot
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d

# Center and width are in eV.
# L3 edge: 708.65
# L2 edge: 721.65
def ArcTanEdge(E, Height, Center, Width=1):
    return Height/np.pi * (np.arctan(np.pi/Width * (E-Center)) + np.pi/2)


def PrepareSpectrum(Energies, GaussianSmooth, PostEdgeEnergy, PreEdgeWindowStart, Spectrum):
    # In order to do all the math goodness below, we need a constant step energy axis.
    # Let's use 10000 steps, just to be wasteful.
    E = np.linspace(Energies[0], Energies[-1], 10000)
    dE = E[1] - E[0]
    S = interp1d(Energies, Spectrum)(E)
    Sraw = S  # Keep this to plot later

    if GaussianSmooth is not None:
        # FWHM is 1, so we want 1/sqrt(8*ln(2)) sigmas in eV.  How many eV/step is dE.
        sigmas = GaussianSmooth / 2.3548 / dE
        S = gaussian_filter1d(S, sigmas)

    # Clip the spectrum to exclude anything before the pre-edge, and anything after the post-edge.
    Sraw = Sraw[E.searchsorted(PreEdgeWindowStart):E.searchsorted(PostEdgeEnergy)]
    S = S[E.searchsorted(PreEdgeWindowStart):E.searchsorted(PostEdgeEnergy)]
    E = E[E.searchsorted(PreEdgeWindowStart):E.searchsorted(PostEdgeEnergy)]
    return E, S, Sraw


def FitPreEdge(E, PreEdgeWindowStart, PreEdgeWindowStop, S):
    # Fit the pre-edge to a line.
    PreStart = E.searchsorted(PreEdgeWindowStart)
    PreStop = E.searchsorted(PreEdgeWindowStop)
    P = np.polyfit(E[PreStart:PreStop], S[PreStart:PreStop], 1)
    Pline = np.poly1d(P)(E)
    return Pline


def FitDoubleArcTanFe(E, L2Amp, L3Amp, Pline, QuietMode, S):
    # Post-edge energy compared against the pre-edge to get the L3 Edge jump.
    L3PostPos = E.searchsorted(715.5)
    # And energy for the L2 Edge jump.
    # L2PostPos = E.searchsorted(727.5)
    L2PostPos = E.searchsorted(730.0)
    # Position of the L3 and L2 edges.
    L3Edge = 708.65
    L2Edge = 721.65

    # Make a background including the arctan edge jumps.
    if (L3Amp is None) and (L2Amp is None):

        TotalAmp = S[L2PostPos] - Pline[L2PostPos]
        L3Amp = 3/5 * TotalAmp
        L2Amp = 2/5 * TotalAmp
        if not QuietMode:
            print('L3 jump computed to be %0.02g' % L3Amp)
            print('L2 jump computed to be %0.02g' % L2Amp)
    else:
        if L3amp is none:
            l3amp = s[l3postpos] - pline[l3postpos]
            if not QuietMode:
                print('L3 jump computed to be %0.02g' % L3Amp)
        if L2Amp is None:
            L2Amp = S[L2PostPos] - Pline[L2PostPos] - L3Amp
            if not QuietMode:
                print('L2 jump computed to be %0.02g' % L2Amp)

    # Compute the spectrum of the edges.
    EdgeL3 = ArcTanEdge(E, L3Amp, L3Edge)
    EdgeL2 = ArcTanEdge(E, L2Amp, L2Edge)
    Edge = EdgeL3 + EdgeL2

    # The total edge jump.
    Jump = L2Amp + L3Amp

    return Edge, Jump

def FitDoubleArcTanTi(E, L2Amp, L3Amp, Pline, QuietMode, S):
    # Define the constants that describe the positions used for the ArcTans.
    # # Post-edge energy compared against the pre-edge to get the L3 Edge jump.
    # L3PostPos = E.searchsorted(461.7)   #optimize this.
    # And energy for the L2 Edge jump.
    L2PostPos = E.searchsorted(468.7)
    # Position of the L3 and L2 edges.
    L3Edge = 457.5
    L2Edge = 463.0

    # Make a background including the arctan edge jumps.
    if L2Amp is None:
        Amp = S[L2PostPos] - Pline[L2PostPos]
        L2Amp = Amp/2
        if not QuietMode:
            print ('L2 jump computed to be %0.02g' % L2Amp)
    if L3Amp is None:
        L3Amp = L2Amp
        if not QuietMode:
            print ('L2 jump assumed equal to L3 jump')

    EdgeL3 = ArcTanEdge(E, L3Amp, L3Edge)
    EdgeL2 = ArcTanEdge(E, L2Amp, L2Edge)
    Edge = EdgeL3 + EdgeL2
    return Edge


def VanAkenMethod1(E, QuietMode, S):
    # Compute van Aken method 1 (least accurate).  Just ratio the intensity in the L3 and L2 lines.
    # L3 intensity is the sum up to 716 eV.  L2 is the sum after.
    L3Intensity = sum(S[:E.searchsorted(716)])
    L2Intensity = sum(S[E.searchsorted(716):])
    L3L2 = L3Intensity / L2Intensity
    # Compute the oxidation state from the ratio.
    # From the van Aken paper, figure 3, the total L3/L2 intensity ratio calibration is a linear fit.
    PL3L2 = [1 / 2.0, -3.5 / 2]
    # Get the value of our ratio from the polynomial.
    Fe3overFe = np.poly1d(PL3L2)(L3L2)
    # Tell the user the answer.
    if not QuietMode:
        print ('Method 1 (accuracy +/- 15%), no consideration for coordination effects')
        print ('I(L3)/I(L2) = %0.2g' % (L3L2))
        print ('Fe3+/sum(Fe) = %0.2g' % (Fe3overFe))
    Method1Fe3overFe = Fe3overFe
    return Method1Fe3overFe


def VanAkenMethod2(E, QuietMode, S):
    # Compute van Aken method 2 (more accurate, using 2 eV energy windows.
    L3Intensity = sum(S[E.searchsorted(708.5):E.searchsorted(710.5)])
    L2Intensity = sum(S[E.searchsorted(719.7):E.searchsorted(721.7)])
    Irat = L3Intensity / L2Intensity
    # This relationship comes from inverting equation 2 in van Aken.
    Fe3overFe = 0.00259067357512953 * (465.0 * Irat - np.sqrt(-(Irat + 1.0) * (66327.0 * Irat - 705673.0)) + 465.0) / (
    Irat + 1.0)
    # Tell the user the answer.
    if not QuietMode:
        print ('Method 2 (accuracy +/- 4%) (2 eV window method)')
        print ('I(L3)/I(L2) = %0.2g' % (Irat))
        print ('Fe3+/sum(Fe) = %0.2g' % (Fe3overFe))
    Method2Fe3overFe = Fe3overFe
    return Method2Fe3overFe


def FeLVanAken(Energies, Spectrum, PreEdgeWindowStart=690, PreEdgeWindowStop=704, PostEdgeEnergy=730,
               GaussianSmooth=0.8, L3Amp=None, L2Amp=None, QuietMode=False):
    # From van Aken, P. A., & Liebscher, B. (2002). Quantification of ferrous/ferric ratios in minerals: new
    # evaluation schemes of Fe L23 electron energy-loss near-edge spectra. Physics and Chemistry of Minerals, 29, 188. doi:10.1007/s00269-001-0222-6

    # Interpolate the spectrum so it has a constant energy step.  Smooth the function to simulate the energy
    # resolution of the Van Aken experiment and clip it to only have the spectrum we care about.
    # van Aken's calibration was done with a LaB6 with 0.8 eV energy resolution.
    E, S, Sraw = PrepareSpectrum(Energies, GaussianSmooth, PostEdgeEnergy, PreEdgeWindowStart, Spectrum)

    # Fit the pre-edge to a line and get the line "spectrum"
    Pline = FitPreEdge(E, PreEdgeWindowStart, PreEdgeWindowStop, S)

    # Fit the L3 and L2 lines using a double arctan.
    Edge, Jump = FitDoubleArcTanFe(E, L2Amp, L3Amp, Pline, QuietMode, S)
    Background = Pline+Edge

    # Show the user.
    if not QuietMode:
        fig, ax = QuickPlot(np.vstack((E,E,E,E)), np.vstack((Sraw, S,Pline, Background)),
                        boldlevel=3,
                        xlabel='eV', title='Smooth spectrum and backgrounds', legendstrs=['Raw','Smoothed','Preedge',
                                                                                       'Preedge+arctans'])

    # Subtract off the background
    S = S-Background

    # Normalize the spectrum to the integral intensity.
    S /= np.sum(S)

    # Get the Fe3+ using Van Aken's method 1 and 2.
    Method1Fe3overFe = VanAkenMethod1(E, QuietMode, S)
    Method2Fe3overFe = VanAkenMethod2(E, QuietMode, S)

    if not QuietMode:
        fig, ax = QuickPlot(np.vstack((E,E)), np.vstack((S, S)), boldlevel=3,
                        xlabel='eV', title='White-line spectrum')
        print (fig, ax)

        plt.show()

    return(Method1Fe3overFe, Method2Fe3overFe)

def FeLStats(Energies, Spectrum, PreEdgeWindowStart=690, PreEdgeWindowStop=704, PostEdgeEnergy=730, L3L2MidEnergy=718.5,
               GaussianSmooth=0.01, L3Amp=None, L2Amp=None, QuietMode=False, ApplyEShift=None, SpectrumName='Spectrum'):

    # Interpolate the spectrum so it has a constant energy step. Smooth the function to 
    # simulate the energy resolution if necessary.
    # Try 0.8 eV energy resolution for TEM.
    E, S, Sraw = PrepareSpectrum(Energies, GaussianSmooth, PostEdgeEnergy, PreEdgeWindowStart, Spectrum)

    # If the user knows he's out of energy calibration then shift the spectrum per his request.
    if ApplyEShift is not None:
        E += ApplyEShift

    # Fit the pre-edge to a line and get the line "spectrum"
    Pline = FitPreEdge(E, PreEdgeWindowStart, PreEdgeWindowStop, S)

    # Fit the L3 and L2 lines using a double arctan.
    Edge, Jump = FitDoubleArcTanFe(E, L2Amp, L3Amp, Pline, QuietMode, S)
    Background = Pline+Edge

    # Show the user.
    fig, ax = QuickPlot(np.vstack((E,E,E,E)), np.vstack((Sraw, S,Pline, Background)),
                    boldlevel=3,
                    xlabel='eV', title='Smooth spectrum and backgrounds', legendstrs=['Raw','Smoothed','Preedge',
                                                                                   'Preedge+arctans'], legendfontscale=0.5)
    plt.savefig(SpectrumName + 'Processing.png', dpi=300)

    # Subtract off the background
    S = S-Background
    SNoBkg = S.copy()

    # Post-edge energy compared against the pre-edge to get the L3 Edge jump.
    L3StartPos = E.searchsorted(PreEdgeWindowStop)
    # Post-edge energy compared against the pre-edge to get the L3 Edge jump.
    L3PostPos = E.searchsorted(L3L2MidEnergy)
    # And energy for the L2 Edge jump.
    L2PostPos = E.searchsorted(PostEdgeEnergy)
    E0Amp = Sraw[L3StartPos]

    # Report the peak/jump ratio.
    print('Maximal peak amplitude is %0.2g' % np.max(S))
    print('Total Jump amplitude is %0.2g' % Jump)
    print('Peak/Jump ratio is %0.2g' %(np.max(S)/Jump))

    # Normalize the spectrum to the edge jump intensity.
    S /= np.sum(Edge)
    SPeaks = S.copy()

    # Now report the area of the edges, and of the L3, L2 lines.
    L3Area = np.sum(S[L3StartPos:L3PostPos])
    L2Area = np.sum(S[L3PostPos:L2PostPos])

    print('Area of the L3 peak is %0.2g' %(L3Area))
    print('Area of the L2 peak is %0.2g' %(L2Area))
    print('L3/L2 area ratio is %0.2g' %(L3Area/L2Area))


    fig, ax = QuickPlot(np.vstack((E,E)), np.vstack((S, S)), boldlevel=3,
                    xlabel='eV', title='White-line spectrum')
    plt.savefig(SpectrumName + 'Peaks Only.png', dpi=300)

    return E, Sraw, SNoBkg, SPeaks, Jump, E0Amp, L3Area, L2Area

def StoyanovMethod(E, QuietMode, S):
    # Integrate the L3 and L2 white lines.
    # L3Intensity = sum(S[E.searchsorted(456.00):E.searchsorted(461.65)])
    # L2Intensity = sum(S[E.searchsorted(461.65):E.searchsorted(467.50)])
    L3Intensity = sum(S[E.searchsorted(455.00):E.searchsorted(460.00)])
    L2Intensity = sum(S[E.searchsorted(460.00):E.searchsorted(465.00)])
    Irat = L2Intensity / L3Intensity
    # This relationship comes from inverting equation 2 in van Aken.
    TiOxidation = 0.21767*np.log((Irat-0.87953)/0.21992)
    # Tell the user the answer.
    if not QuietMode:
        print ('Stoyanov method - I(L2)/I(L3)')
        print ('I(L2)/I(L3) = %0.2g' % (Irat))
        print ('Ti4+ fraction = %0.2g' % (TiOxidation))
    return TiOxidation

def TiLStoyanov(Energies, Spectrum, PreEdgeWindowStart=447.5, PreEdgeWindowStop=454.0, PostEdgeEnergy=482,
               GaussianSmooth=0.8, L3Amp=None, L2Amp=None, QuietMode=False):
    # From Stoyanov, E., Langenhorst, F., & Steinle-Neumann, G. (2007). The effect of valence state and site geometry
    # on Ti L3,2 and O K electron energy-loss spectra of TixOy phases. American Mineralogist, 92(4), 577-586.
    # doi:10.2138/am.2007.2344

    # Interpolate the spectrum so it has a constant energy step.  Smooth the function to simulate the energy
    # resolution of the Van Aken experiment and clip it to only have the spectrum we care about.
    # van Aken's calibration was done with a LaB6 with 0.8 eV energy resolution.
    E, S, Sraw = PrepareSpectrum(Energies, GaussianSmooth, PostEdgeEnergy, PreEdgeWindowStart, Spectrum)

    # Fit the pre-edge to a line and get the line "spectrum"
    Pline = FitPreEdge(E, PreEdgeWindowStart, PreEdgeWindowStop, S)

    # Fit the L3 and L2 lines using a double arctan.
    Edge = FitDoubleArcTanTi(E, L2Amp, L3Amp, Pline, QuietMode, S)
    Background = Pline+Edge

    # Show the user.
    if not QuietMode:
        fig, ax = QuickPlot(np.vstack((E,E,E,E)), np.vstack((Sraw, S,Pline, Background)),
                        boldlevel=3,
                        xlabel='eV', title='Smooth spectrum and backgrounds', legendstrs=['Raw','Smoothed','Preedge',
                                                                                       'Preedge+arctans'])

    # Subtract off the background
    S = S-Background

    # Normalize the spectrum to the integral intensity.
    S /= np.sum(S)

    # Get the Ti oxidation state using the part the hasn't been programmed yet...
    OxidationState = StoyanovMethod(E, False, S)

    if not QuietMode:
        fig, ax = QuickPlot(np.vstack((E,E)), np.vstack((S, S)), boldlevel=3,
                        xlabel='eV', title='White-line spectrum')
        print (fig, ax)

        plt.show()

    return(OxidationState)

if __name__ == '__main__':

    E = np.genfromtxt('532_190625016_energies.txt')
    Mickey = np.genfromtxt('Spectrum 1 Mickey.txt', skip_header=1)[:,1]
    Sulfide = np.genfromtxt('Spectrum 2 Sulfide.txt', skip_header=1)[:,1]
    Alpha = np.genfromtxt('Spectrum 3 Alpha.txt', skip_header=1)[:,1]
    Beta = np.genfromtxt('Spectrum 4 Beta.txt', skip_header=1)[:,1]
    Gamma = np.genfromtxt('Spectrum 5 Gamma.txt', skip_header=1)[:,1]
    Silicate = np.genfromtxt('Spectrum 6 Silicate.txt', skip_header=1)[:,1]

    def ShowMeSpectrum(SpectrumName='', Spectrum=None):
        print(f'----- {SpectrumName} -----')
        SpectrumProc = FeLStats(E, Spectrum, SpectrumName=SpectrumName + ' ', GaussianSmooth=0.1)
        os.system(f'/Users/Zack/.iterm2/imgcat "{SpectrumName} Processing.png"')
        os.system(f'/Users/Zack/.iterm2/imgcat "{SpectrumName} Peaks Only.png"')
        return SpectrumProc

    MickeyProc = ShowMeSpectrum('Spectrum 1 Mickey', Mickey)
    SulfideProc = ShowMeSpectrum('Spectrum 2 Sulfide', Sulfide)
    AlphaProc = ShowMeSpectrum('Spectrum 3 Alpha', Alpha)
    BetaProc = ShowMeSpectrum('Spectrum 4 Beta', Beta)
    GammaProc = ShowMeSpectrum('Spectrum 5 Gamma', Gamma)
    SilicateProc = ShowMeSpectrum('Spectrum 6 Silicate', Silicate)

    fig, ax = QuickPlot(SulfideProc[0], 3.5*(SulfideProc[1]-np.min(SulfideProc[1])), boldlevel=3)
    QuickPlot(MickeyProc[0], 30*(MickeyProc[1]-np.min(MickeyProc[1]))+2, boldlevel=3, figax=(fig,ax))
    QuickPlot(AlphaProc[0], 30*(AlphaProc[1]-np.min(AlphaProc[1]))+4, boldlevel=3, figax=(fig,ax))
    QuickPlot(BetaProc[0], 24*(BetaProc[1]-np.min(BetaProc[1]))+6, boldlevel=3, figax=(fig,ax))
    QuickPlot(GammaProc[0], 80*(GammaProc[1]-np.min(GammaProc[1]))+8, boldlevel=3, figax=(fig,ax))
    QuickPlot(SilicateProc[0], 40*(SilicateProc[1]-np.min(SilicateProc[1]))+10, boldlevel=2, legendfontscale=0.5, figax=(fig,ax), xlabel='eV', ylabel='O.D. A.U.', legendstrs=['Sulfide', 'Mickey', 'Alpha', 'Beta', 'Gamma', 'Silicate'], xlim=[700.0, 730.0])
    plt.savefig('AllSpectra.png', dpi=300)
    os.system(f'/Users/Zack/.iterm2/imgcat "AllSpectra.png"')
    plt.show()

    # Additional examples.

    ## Test Ti.
    #LoadedSpectrum = np.genfromtxt('/Users/Stardust/Zack/MoonBaseBeta/ParticleBase/Track 162a - Cecil/Grid A6/20131112 - STXM - Cecil A6, S3/131113/11_131113012 - Ti stack/Fiji/11_131113012_Spectrum1 bkgremoved.txt', comments = '%')
    #Energies = LoadedSpectrum[:,0]
    #Spectrum = LoadedSpectrum[:,1]
    #TiLStoyanov(Energies, Spectrum)

    ## Now test Fe.

    #Energies = np.genfromtxt('/Users/Stardust/Zack/MoonBaseBeta/ParticleBase/Track 162a - Cecil/Grid A6/20131121 - STXM - Cecil A6, S3/11_131121017_Cecil_a6s3_Fe/Components/Energies.txt')
    #Spectrum = np.genfromtxt('/Users/Stardust/Zack/MoonBaseBeta/ParticleBase/Track 162a - Cecil/Grid A6/20131121 - '
    #                         'STXM - Cecil A6, S3/11_131121017_Cecil_a6s3_Fe/Components/Spectrum - Spinel.txt', usecols=1)
    #Energies = Energies
    #A = np.linspace(0,0.05, 50)
    #x = []
    #for a in A:
    #    print (a)
    #    x.append(FeLVanAken(Energies, Spectrum, L3Amp=a, QuietMode=True))

    #Method1, Method2 = zip(*x)

    ##plt.clear('all')
    #plt.plot(A, Method1, A, Method2)
    #plt.xlabel('L3Amp')
    #plt.ylabel('Fe3+/sum(Fe)')
    #plt.legend(['Method 1', 'Method 2'])
    #plt.show(block=False)

    #FeLVanAken(Energies, Spectrum)
    #print ('Done')
