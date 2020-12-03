import numpy as np
import glob
import starlink
#starlink.wrapper.change_starpath("/stardev")
from starlink import kappa

def apply_relFCF_AC(tablefile,wave):

    table = np.genfromtxt(tablefile,dtype=None,names=True)
    region = tablefile.split('_')[1]

    FCFkey = 'CalF_gauss'

    for eachdate,eachFCF in zip(table['Key'],table[FCFkey]):
        eachdatestr = eachdate.decode('utf-8')
        kappa.cdiv(region+'_'+eachdatestr.split('-')[0]+'_'+eachdatestr.split('-')[1].zfill(5)+'_'+wave+'_CR3_crop.sdf',eachFCF,region+'_'+eachdatestr.split('-')[0]+'_'+eachdatestr.split('-')[1].zfill(5)+'_'+wave+'_CR3_ACcal.sdf')

