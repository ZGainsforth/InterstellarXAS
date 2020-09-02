import numpy as np
import os, sys
sys.path.append(os.path.abspath('..'))
import InterstellarXASTools

SubDirName = 'XMM Spectra'

config = InterstellarXASTools.init_config()

def SaveOneSpectrum(SourceName, obsid):
    angstromsum, eVsum, fluxsum, errorsum, angstrom_label, eV_label, flux_label, error_label = InterstellarXASTools.GetOneXMMSpectrum(config, obsid)

    # Make sure the directory for this source name exists.
    if not os.path.exists(os.path.join(SubDirName, SourceName)):
        os.mkdir(os.path.join(SubDirName, SourceName))

    # Convert the flux units from photons/(cm2 s A) to photons/(cm2 s keV)
    assert flux_label == '1/(s cm^2 A)', "Incorrect units for flux density."
    fluxsum = fluxsum / 12.4 * angstromsum**2
    flux_label  = '1/(s cm^2 keV)'
    assert error_label == '1/(s cm^2 A)', "Incorrect units for flux density."
    errorsum = errorsum / 12.4 * angstromsum**2
    error_label = '1/(s cm^2 keV)'
    
    FileName = f'{obsid}.csv'
    with open(os.path.join(SubDirName, SourceName, FileName), 'w') as f:
        keV_label = 'keV'
        f.write(f'# {keV_label:>19s}, flux in {flux_label:>12s}, flux error in {error_label:>7s},   counts, {angstrom_label:>20s}, {eV_label:>21s}\n')
        for i in reversed(range(len(eVsum))):
            f.write(f' {eVsum[i]/1000:21f}    {fluxsum[i]:20f}    {errorsum[i]:26f}  {1:6f}  {angstromsum[i]:20f}  {eVsum[i]:20f}\n')
    print(f'Wrote {os.path.join(SubDirName, SourceName, FileName)}')
    
SourceObsIDs = {
        '4U 1636-536': [303250201, 500350301, 500350401, 606070101, 606070201, 606070401, 606070301, 764180201, 764180301, 764180401],
              
        '4U1735-444': [693490201, 90340201],

        'Aql X-1': [743520201, 67750801, 67751001, 85180101, 85180201, 85180301, 85180401, 85180501, 112440101, 112440301, 112440401, 303220201, 303220301, 303220401, 406700201],

        'Cyg X-1': [202400501, 202400701, 202760201, 202760301, 202760401, 202760501, 605610401, 745250701, 202401101, 202401201, 745250201, 745250501, 745250601],

        'Cyg X-2': [111360101, 303280101, 561180201, 561180501],

        'GRO J1655-40' : [112460201, 112460301, 112921301, 112921401, 112921501, 112921601, 155762501, 155762601, 400890201, 400890301],

        'GS 1354-645': [727961501],

        'GS 1826-238': [150390101, 150390301, 156160101],

        'GX 339-4': [148220201, 204730201, 204730301, 605610201, 654130401, 760646201, 760646301, 760646401, 760646501, 760646601, 760646701],

        'SAX J1808.4-3658': [400230401, 400230501, 724490201, 560180601, 64940101],

        'Serpens X-1': [84020401, 84020501, 84020601], 

        'Swift J1753.5-0127': [311590901, 605610301, 744320201, 744320301, 770580201, 694930501, 691740201], 

        'Swift J1910.2-0546': [741590101, 691271401], 

        'XTE J1650-500': [136140301, 206640101]
        }

for k, v in SourceObsIDs.items():

    for obsid in v:
        try:
            SaveOneSpectrum(k, str(obsid))
        except:
            print(f'Skipping {obsid}.')
    print('\n')







