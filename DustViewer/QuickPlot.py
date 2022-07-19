# Created 2015, Zack Gainsforth
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from itertools import chain

def QuickPlot(x, y, linestyle='', xlabel=None, ylabel=None, title=None, legendstrs=None, legendloc=None, savefile=None, boldlevel=1, figax=None, plottype=None, xlim=None, ylim=None, figsize=None, legendfontscale=None, **kwargs):
    """
    :param x: Can be a 1D numpy array, or a 2D numpy array.  If 1D, then there is one plot.  If 2D, every row is the x-axis for another line on the plot.
    :param y: Identical dimension to x.
    :param xlabel: String (can include TeX) for the label of the x-axis.
    :param ylabel: String (can include TeX) for the label of the y-axis.
    :param title: String (can include TeX) for the label of the title.
    :param legendstrs: List of strings with the same length as x and y, axis 1.
    :param savefile: A filename to save to if desired.
    :param boldlevel: Sets the linewidths fatter and texts bigger.  1 is normal, 4 is usually good for publication.
    :param figax: None makes a new figure and axis.  Or plots on an existing one if a tuple is given: (fig, ax)
    :return: (fig, ax) The figure and axis so the user can do stuff of his own with it if he wants to.
    """

    # Set the bold level.
    FontSizeBasis = (boldlevel+2)*4    # Fonts get bigger as boldlevel increases
    TickMajorBasis = boldlevel*4    # As fonts get bigger, they need a larger padding from the axis.
    # Increase the size of the tick label fonts.
    matplotlib.rc('xtick', labelsize=FontSizeBasis)
    matplotlib.rc('ytick', labelsize=FontSizeBasis)
    # Increase their padding.
    matplotlib.rc('xtick.major', pad=TickMajorBasis)
    matplotlib.rc('ytick.major', pad=TickMajorBasis)

    # Create/reuse a figure to plot on.
    if figax is None:
        if figsize is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = figax

    # Figure out how many lines are going on this plot.
    dim = np.shape(x)
    if len(dim) == 1:
        # Just one line on this plot.
        if plottype is None:
        	ax.plot(x,y, linestyle, markersize=TickMajorBasis, linewidth=boldlevel, **kwargs)
        elif plottype == 'scatter':
        	ax.scatter(x,y, s=TickMajorBasis**2, **kwargs)
        elif plottype == 'semilogy':
        	ax.semilogy(x,y, linestyle, markersize=TickMajorBasis, linewidth=boldlevel, **kwargs)
        elif plottype == 'semilogx':
        	ax.semilogx(x,y, linestyle, markersize=TickMajorBasis, linewidth=boldlevel, **kwargs)
        else: # plottype == 'loglog':
        	ax.loglog(x,y, linestyle, markersize=TickMajorBasis, linewidth=boldlevel, **kwargs)
        
    elif len(dim) == 2:
        # Multiple lines on this plot.
        # For multiple lines we cannot use linestyle.
        for i in range(dim[0]):
	        if plottype is None:
	        	ax.plot(x[i,:], y[i,:], linewidth=boldlevel, **kwargs)
        	elif plottype == 'scatter':
        		ax.scatter(x,y, s=TickMajorBasis**2, **kwargs)
	        elif plottype == 'semilogy':
	        	ax.semilogy(x[i,:], y[i,:], linewidth=boldlevel, **kwargs)
	        elif plottype == 'semilogx':
	        	ax.semilogx(x[i,:], y[i,:], linewidth=boldlevel, **kwargs)
	        else: # plottype == 'loglog':
	        	ax.loglog(x[i,:], y[i,:], linewidth=boldlevel, **kwargs)
    else:
        raise TypeError('x is not a 1D or 2D numpy array.')

    # Figure out how many lines are going on this plot.
    if 'lines' in ax.__dir__():
        for line in ax.lines:
            line.set_linewidth(boldlevel)
    if 'markers' in ax.__dir__():
        for marker in ax.markers():
            marker.set_markersize(TickMajorBasis)
    if 'points' in ax.__dir__():
        for point in ax.points():
            point.set_s(TickMajorBasis**2)
    ax.set_xlabel(ax.get_xlabel(), fontsize=FontSizeBasis)
    ax.set_ylabel(ax.get_ylabel(), fontsize=FontSizeBasis)
    ax.set_title(ax.get_title(), fontsize=FontSizeBasis)
    ax.tick_params(axis='both', which='major', labelsize=FontSizeBasis)
    ax.tick_params(axis='both', which='minor', labelsize=FontSizeBasis)

    # Write the x and y labels and the title.
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FontSizeBasis)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FontSizeBasis)
    if title is not None:
        ax.set_title(title, fontsize=FontSizeBasis)

    # Show a legend if input.
    if legendstrs is not None:
        if legendfontscale is None:
            ax.legend(legendstrs, fontsize=FontSizeBasis, loc=legendloc)
        else:
            ax.legend(legendstrs, fontsize=FontSizeBasis*legendfontscale, loc=legendloc)

    # Set the x and y display ranges
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Resize the figure appropriately
    try:
    	plt.draw()
    	plt.tight_layout()
    except:
    	print('plt.tight_layout() complained ... again ...')

    # Save it to a file if a name is given.
    if savefile is not None:
        plt.savefig(savefile)

    return fig, ax

def PrettyPlot(figax, linestyle='', xlabel=None, ylabel=None, title=None, legendstrs=None, legendloc=None, savefile=None, boldlevel=1, plottype=None, xlim=None, ylim=None, figsize=None, legendfontscale=None, **kwargs):
    """
    :param x: Can be a 1D numpy array, or a 2D numpy array.  If 1D, then there is one plot.  If 2D, every row is the x-axis for another line on the plot.
    :param y: Identical dimension to x.
    :param xlabel: String (can include TeX) for the label of the x-axis.
    :param ylabel: String (can include TeX) for the label of the y-axis.
    :param title: String (can include TeX) for the label of the title.
    :param legendstrs: List of strings with the same length as x and y, axis 1.
    :param savefile: A filename to save to if desired.
    :param boldlevel: Sets the linewidths fatter and texts bigger.  1 is normal, 4 is usually good for publication.
    :param figax: None makes a new figure and axis.  Or plots on an existing one if a tuple is given: (fig, ax)
    :return: (fig, ax) The figure and axis so the user can do stuff of his own with it if he wants to.
    """

    # Set the bold level.
    FontSizeBasis = (boldlevel+2)*4    # Fonts get bigger as boldlevel increases
    TickMajorBasis = boldlevel*4    # As fonts get bigger, they need a larger padding from the axis.
    # Increase the size of the tick label fonts.
    matplotlib.rc('xtick', labelsize=FontSizeBasis)
    matplotlib.rc('ytick', labelsize=FontSizeBasis)
    # Increase their padding.
    matplotlib.rc('xtick.major', pad=TickMajorBasis)
    matplotlib.rc('ytick.major', pad=TickMajorBasis)

    # Create/reuse a figure to plot on.
    assert figax is not None, "fig and ax must be specified in order to prettyify the plot."

    fig = figax[0]
    ax = figax[1]

    # Figure out how many lines are going on this plot.
    if 'lines' in ax.__dir__():
        for line in ax.lines:
            line.set_linewidth(boldlevel)
    if 'markers' in ax.__dir__():
        for marker in ax.markers():
            marker.set_markersize(TickMajorBasis)
    if 'points' in ax.__dir__():
        for point in ax.points():
            point.set_s(TickMajorBasis**2)
    ax.set_xlabel(ax.get_xlabel(), fontsize=FontSizeBasis)
    ax.set_ylabel(ax.get_ylabel(), fontsize=FontSizeBasis)
    ax.set_title(ax.get_title(), fontsize=FontSizeBasis)
    ax.tick_params(axis='both', which='major', labelsize=FontSizeBasis)
    ax.tick_params(axis='both', which='minor', labelsize=FontSizeBasis)

    # Show a legend if input.
    if legendstrs is not None:
        ax.legend(legendstrs)
    if legendloc is not None:
        txts = []
        for t in ax.get_legend().get_texts():
            txts.append(t.get_text())
        plt.legend(txts, loc=legendloc)
    if ax.get_legend() is not None:
        if legendfontscale is None:
            for txt in ax.get_legend().get_texts():
                txt.set_fontsize(FontSizeBasis)
        else:
            for txt in ax.get_legend().get_texts():
                txt.set_fontsize(FontSizeBasis*legendfontscale)
        for l in ax.get_legend().get_lines():
            l.set_linewidth(boldlevel)

    # Set the x and y display ranges
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Resize the figure appropriately
    try:
    	plt.draw()
    	plt.tight_layout()
    except:
    	print('plt.tight_layout() complained ... again ...')

    # Save it to a file if a name is given.
    if savefile is not None:
        plt.savefig(savefile)

    return fig, ax

# About 2/3 of my plots are just 2-column data.  So make it easy to plot these.
def SpecPlot(S, DivBkg=None, SubBkg=None, **kwargs):
    
    # Make E (xaxis) and I (yaxis)
    E = np.copy(S[:,0])
    I = np.copy(S[:,1])
    
    # Divide by a background spectrum.
    if DivBkg is not None:
        DivNFunc = interp1d(DivBkg[:,0], DivBkg[:,1], bounds_error=False, fill_value=0)
        DivN = DivNFunc(S[:,0])
        I /= DivN
        
    if SubBkg is not None:
        SubNFunc = interp1d(SubBkg[:,0], SubBkg[:,1], bounds_error=False, fill_value=0)
        SubN = SubNFunc(S[:,0])
        I -= SubN
        E = S[:,0]

    NewSpectrum = np.vstack((E, I)).T

    fig, ax = QuickPlot(E, I, **kwargs)
    
    return(fig, ax, NewSpectrum)
