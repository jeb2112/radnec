# quick script to download files from dbox
# using 'nnunet' env

import dropbox
import os
import numpy as np
import struct
import nibabel as nb

# manually pre-generate new short lived token to use this script
# there should be a way to do this by program with oauth or something.
token = 'sl.u.AFgc3IoMHqqEsWJOEqqFuB3RLEUpwARsVtdZKmMvkkbRFPiCqrmfg_GhaWPTBcUbu1YK6Scfo5TEVwmq6jsjk3fYtqtnoyZ8cVReJc3fKQawlIppPqjBE4nUhPuwd-40X9CMMn4DuZYmprWXw4CREZxQMahWUMcZ2vdVvM_ugYNq28ZatqzGpGRfsqwfndolY8O2rmeUCmgPZ2vaHlwRO7Y9r6ZUGAN-hF3krEAKHY2VBYXbFfLBnyoEGOIBj8pxAgPNrsnx6-cB-9HLHz8QQOvotpE7gDSUC9ejsn_ijtq50fYLjhDGPUNkzAjQiaQqeu492kNHBNXfOHmTGdgwH5np6Q7KnlmKTKst0EBwYeZQrEeBAnp_Wp6MepovoP1_nNuWBrRWF-2ZCE6BgZjXqkiPCaf3-6wjZC5vUPdIk7rCH8d40DSbDsE5Ga5cEmUhU52sAij3pmvgdcLBUai82hCYDJ2UdrWMPIKSBd96icxjh0rhIo6ZolZStlYLIq-1raFkocK3ADfPp4J_CxHsEte-hlBNXxp5Em_5coGYZyvBok87jWcbBEuTli4Fgm02uv8qhQex7N3gCZee6LyhBnslEQSAxDwAYqtHfwH37zQmxhlihERJ-HJZ_e8zTE8_UotOyK3dQXHhg_SxsetSbhwZWwsMkH5qbbGoms0dqFMEA0C0EG7qDcpuOjFwWojeqpXKwgKayTltaDcsFkTsRyw02IEQTMEvwbtL-UCKCVAWQBL_7jNjKQgS1hoB6kXxhGVSeCMR0nVIMftmBcd0EdyHoI79B8SQ73ils-QQjabiLburY6u_4ziLOdTYPG4sZzIX2W1j_C6FDBaYdI3sbSF-HYRlq0uzYewl2JY7Z-YbYIzCyrXJikJboEYRdGB_EMhLBU1ZDCpXswqD1zQSwCnbzvHrX0AgsevzHyQSKgYT0eL8-ZI9n0V4NWGs2ADGvqzufLd2JtmBErKZPhY8nIGWoF8BHdQjA3hE6kqTbE01Sw4Xj27veZd_WfGvST2xPal3KDRwTR-BsxZov34jvxFbDR0xqWv0xBNGFSRsaJVIaeh_ibHjsjUNplSm0WXKlVaNQMzola-QesjqQXIbGieuPVuXpqpaAiJjBFYGMA3lZILrOOPU6Mileu77KtMgDeXEyXd3dE5eh9jnaMTikBkASv3IP3CHofyZb6Xv9LMl7026z7CwLi8nENDPpd7h8I5khzfuktDgpC1yRPQ24XLImTQdZ-vzbjcIhhADKVYSbj1vE3ynNSeZTqKT8V_1BZa5sxKRfjgIwRCB6YFWOCXT'

dbox = dropbox.Dropbox(token)
a = dbox.users_get_current_account()

rootdir = '/RADOGEN/R&D/DATA/RAD NEC'

datadir = os.path.join(rootdir,'SEGMENTATIONS')

res = dbox.files_list_folder(datadir)
rv = {}
for entry in res.entries:
    if entry.name.startswith('M') or entry.name.startswith('DSC_'):
        rv[entry.name] = entry

localdir = '/media/jbishop/WD4/brainmets/sunnybrook/radnec2'
segdir = os.path.join(localdir,'seg')

flist = ['mask_T.nii','mask_TC.nii','t1+_processed.nii.gz','flair+_processed.nii.gz']
for c in rv.keys():
    print(c)
    if False:
        if c != 'M0003':
            continue
    casedir = os.path.join(datadir,c)

    # study dirs. for now, only 1 study is segmented
    res = dbox.files_list_folder(casedir)

    for rdir in res.entries:
        
        sdir = os.path.join(casedir,rdir.name)
        res2 = dbox.files_list_folder(sdir)
        rlist = [x.name for x in res2.entries]
        # mask_T is not present if no tumor
        if set(flist[1:]).issubset(set(rlist)):

            # for individual files
            for f in flist:
                try:
                    md,res3 = dbox.files_download(os.path.join(sdir,f))
                except dropbox.exceptions.HttpError as e:
                    print('HTTP error',e)
                    print('skipping {}'.format(f))
                    continue
                except ConnectionError as e:
                    print('HTTP error',e)
                    print('attempt to re-connect')
                    # this isn't global scope though
                    dbox = dropbox.Dropbox(token)
                    md,res = dbox.files_download(os.path.join(casedir,f))
                    continue
                except dropbox.exceptions.ApiError as e:
                    if 'not_found' in str(e.error):
                        print('{} not found, skipping'.format(f))
                    else:
                        print('API error',e.error)
                    continue

                local_casedir = os.path.join(segdir,c,rdir.name)
                if not os.path.exists(local_casedir):
                    os.makedirs(local_casedir)
                fname = os.path.join(local_casedir,f)
                with open(fname,'wb') as fp:
                    fp.write(res3.content)
                    if 'mask' in f:
                        os.system('gzip --force "{}"'.format(fname))
            break
        elif rdir == res.entries[-1]:
            print('no match for case {}'.format(c))
    else:
        continue
    continue
 

if False:
    for entry in dbox.files_list_folder(rootdir).entries:
        print(entry.name)

a=1
