import numpy as np
from numpy.lib import recfunctions
import math

import astropy.io.fits as fits
import astropy.coordinates
from astropy import units as u
from astropy.utils.console import ProgressBar
APO_location = astropy.coordinates.EarthLocation("-105:49:13d", "32:46:49d")

def process_data(files, outfile):
    # Open the 20 individual files corresponding to different airmass selection criteria
    print "Reading data"
    data = np.hstack([fits.open(f)[1].data for f in files])
    # Add useful new columns to the recarray
    # LMST = local mean sidereal time [hours]
    # HA = hour angle [hours]
    # za = zenith angle [degrees]
    # q = parallactic angle = position angle of the zenith from the object 
    #     measured E from N [degrees]
    # e_plus = ellipticity par/perp to alt/az coords
    # e_cross = ellipticity 45-deg rot to alt/az coords
    # ePSF... = same for the PSF model (as opposed to the raw stellar measurements)
    # Ixx = Second moment along altitude direction
    # Iyy = Second moment along azimuth direction
    # Ixy = Second moment along altitude=azimuth direction [square pixels]
    # I??PSF = same for the PSF model [square pixels]
    # dalt/daz = astrometric offset relative to r-band in alt/az coords [degrees]

    print "Appending new fields to data array"
    new_fields = []
    for f in 'ugriz':
        new_fields.extend([s+'_'+f for s in
                       ['ra', 'dec', 'LMST', 'HA', 'za', 'q',
                        'e_plus', 'e_cross', 'ePSF_plus', 'ePSF_cross',
                        'I_xx', 'I_yy', 'I_xy', 'IPSF_xx', 'IPSF_yy', 'IPSF_xy',
                        'dalt', 'daz']])
    data = recfunctions.append_fields(data, new_fields,
                                      data = np.ones((len(new_fields), len(data)),
                                                     dtype=float)*np.nan,
                                      usemask=False, asrecarray=True)

    # Fill in new fields
    print "Constructing per-band RA and dec"
    for f in 'ugriz':
        # Note that 'ra' is in degrees, but offsetRa is in arcsec
        # Also, offsetRa is measured as the great circle angle between the r-band centroid
        # and the centroid of the band in question, not the polar angle (i.e., not the difference
        # between the ra at one point and the ra at another point).  So there's a cos(delta) 
        # factor that must be included.
        data['ra_'+f] = data['ra']+data['offsetRa_'+f] / 3600. / np.cos(data['dec']*np.pi/180)
        data['dec_'+f] = data['dec']+data['offsetDec_'+f] / 3600.0

    print "Computing Local Mean Sidereal Time and Hour Angle"
    for f in 'ugriz':
        t = astropy.time.Time(data['TAI_'+f] / (24*3600), format='mjd', location=APO_location)
        lmst = t.sidereal_time('mean')
        data['LMST_'+f] = lmst.hourangle
        ha = data['LMST_'+f] - data['ra_'+f]/15.0 # store HA in hours
        # wrap to between -12 and +12 hours
        ha[ha < -12] += 24.0
        ha[ha > 12] -= 24.0
        data['HA_'+f] = ha

    # It's easy to get your angles mixed up, but thankfully the airmass can both be computed 
    # via the derived zenith angle, and is also a column in the database itself, which allows for
    # some sanity checking.  Unfortunately, it's much harder to sanity check the parallactic angle.
    print "Computing zenith and parallactic angles"
    for f in 'ugriz':
        zenith_coord = astropy.coordinates.SkyCoord(ra=data['LMST_'+f]*u.hourangle,
                                                    dec=APO_location.latitude)
        obj_coord = astropy.coordinates.SkyCoord(ra=data['ra_'+f]*u.deg, dec=data['dec_'+f]*u.deg)
        zenith_angle = zenith_coord.separation(obj_coord)
        data['za_'+f] = zenith_angle.deg
        q = obj_coord.position_angle(zenith_coord).wrap_at(180*u.deg)
        data['q_'+f] = q.deg

    print "Computing differential alt/az"
    for f in 'ugriz':
        data['dalt_'+f] = (np.cos(data['q_'+f]*np.pi/180) * data['offsetDec_'+f]
                           + np.sin(data['q_'+f]*np.pi/180) * data['offsetRa_'+f])
        data['daz_'+f] = (-np.cos(data['q_'+f]*np.pi/180) * data['offsetRa_'+f]
                           + np.sin(data['q_'+f]*np.pi/180) * data['offsetDec_'+f])

    print "Rotating ellipticities to alt/az frame"
    with ProgressBar(len(data)) as bar:
        # might be possible to parallize this better, but I didn't bother
        for i, d in enumerate(data):
            bar.update()
            for f in 'ugriz':
                phi = (d['phioffset_'+f] + d['q_'+f]) * math.pi/180
                R = np.matrix([[math.cos(2*phi), math.sin(2*phi)],
                               [-math.sin(2*phi), math.cos(2*phi)]])
                e = R * np.matrix([d['mE1_'+f], d['mE2_'+f]]).T
                data['e_plus_'+f][i] = e[0]
                data['e_cross_'+f][i] = e[1]
                ePSF = R * np.matrix([d['mE1PSF_'+f], d['mE2PSF_'+f]]).T
                data['ePSF_plus_'+f][i] = ePSF[0]
                data['ePSF_cross_'+f][i] = ePSF[1]

    print "Computing second moments"
    for f in 'ugriz':
        data['I_xx_'+f] = 0.5*data['mRrCc_'+f]*(1.0+data['e_plus_'+f])
        data['I_yy_'+f] = 0.5*data['mRrCc_'+f]*(1.0-data['e_plus_'+f])
        data['I_xy_'+f] = 0.5*data['mRrCc_'+f]*data['e_cross_'+f]
        data['IPSF_xx_'+f] = 0.5*data['mRrCcPSF_'+f]*(1.0+data['ePSF_plus_'+f])
        data['IPSF_yy_'+f] = 0.5*data['mRrCcPSF_'+f]*(1.0-data['ePSF_plus_'+f])
        data['IPSF_xy_'+f] = 0.5*data['mRrCcPSF_'+f]*data['ePSF_cross_'+f]

    np.save(outfile, data)

if __name__ == '__main__':
    # import glob
    # zenith_files = glob.glob("StarShapeX??????_jmeyers3.fit")
    # process_data(zenith_files, "data_uniform_zenith_angle.npy")

    # airmass_files = ["StarShapeX{0}_jmeyers3.fit".format(X) for X in xrange(105, 201, 5)]
    # process_data(airmass_files, "data_uniform_airmass.npy")

    zenith1_files = ["StarShapeZenith_jmeyers3.fit"]
    process_data(zenith1_files, "data_zenith.npy")