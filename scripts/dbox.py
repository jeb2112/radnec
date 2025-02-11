# quick script to download files from dbox
# using 'nnunet' env

import dropbox
import os
import numpy as np
import struct
import nibabel as nb

# manually pre-generate new short lived token to use this script
# there should be a way to do this by program with oauth or something.
token = 'sl.u.AFijc01qpSdQXjhLwXSfXabJAShn5q3IsKen6626fVAW8szEG9Mj3kbTp0qwk1sUm5fpYUfdkVrtX8RDKE1KRNtmCoL5a0N6Uow4CD_sTNtNgzZ7qCDr_h-Xi4UySfWhZWKVaI6zEbadW17QxME-Qdn6avIym2Nj8xzd2PGKb0Iw_HLMkK_kCxxT87_EqRtfteYb2-6XM184Wzed31QfTXE0UgT0qu3XNXN2a6gfaGLlwTaY5InR-ZcpwDX6Rs6AXtvUesNb4CyFwzGaAeRjhDWbVF1cc9bEYpppKHUtt9vZ8nq8hzR24lERXjXb18Hc0-u06ExJkUhQt_G7KCgMSw2wTo3IcDl6zZ9u4tk5Lo6DrezfJqYzBUIf7arbnk5xiLxTDH2gFOWGzTCDz9OqJAioqCt0YCgasCAw2thvF5RTTZqRkuuOpEk8WHG6JLghLDkY7fsd73jQr9jbpyPKfbdGxFSpdziiDxuRVxtLoOjzfZSIEWlAraVme0K7loTBhRhPbny1IEnMKL6UAdDQ-pL_8iIt_c2O_CJCGXYom04psaehgSKjSYjwiIlzO8kcFk5hvQSVwmAGFJboJTRD4OS6z5B9ClCg_tqW1pj1KjJOtyabadSyxEr8PnGswQXm3PLo1I2TG5ah23D6kwNgaH3jLh5LvcesN9Jhn50K2ZbRRxRram7iIZpn2Gzd19cZgJq6tiAmkgKfjbSNUK45wK1Uynn18ekgGdGN40gw-MEoTJud4fGiK5ACz6k1mkcJrMM1VQE3MMBGXKPg8xma_yIfwnsohi0zCXF_kRgp0QPnuCgsCCjs8UYeAnN3ZZjWcLA78GkS2du9bLmjPLal9EIUYDMG0e-ky-_9OPIMERNemTK5um8_t0VRHYIlW_qIot_EHoO-u7aWKzJa4T7IxGb-gs_VxDmdrXt9QKPdNPme3sWLtdOThk6U8SODu-_h1TB0yOhhuxTF4m9xNvo28gJmnjzZklZkMOkoomiiLrdLFUN1FXM4kBrWZmUxwGLjcgXeRTlsIt_tnWGZ0efwww_wsmKwTNs1sybhHMwUes0YLT2bZTecqkI8hSs-4_8-Gau_KBuk_N-x_TNR8lF8a9IwG5xdl73ncsYtsxk-29bMN1VpPtJZl2NwYoqecxrpbJCkR9SUG5-AbUAPGYf86tq3Q81Oj52Vl8Y5I70sSDTbiijPnT8Yx6luubH9SLSbOlNVRwGtHZtmNWGVTUpOGmkof4QIeZQXBwJnM_i-jDJ-r_t2A1R1dGm-3etuhD5oDsabNcqMTON3Wz9C_aJXIUI0'

dbox = dropbox.Dropbox(token)
a = dbox.users_get_current_account()

rootdir = '/RADOGEN/R&D/DATA/RAD NEC'
datadir = rootdir + '/SEGMENTATIONS'

res = dbox.files_list_folder(datadir)
rv = {}
for entry in res.entries:
    if entry.name.startswith('M') or entry.name.startswith('DSC_'):
        rv[entry.name] = entry

if os.name == 'posix':
    localdir = '/media/jbishop/WD4/brainmets/sunnybrook/radnec2'
else:
    localdir = os.path.join('D:','data','radnec2')
segdir = os.path.join(localdir,'seg')

flist = ['mask_T.nii','mask_TC.nii','t1+_processed.nii.gz','flair+_processed.nii.gz']
for c in rv.keys():
    print(c)
    if False:
        if c != 'M0003':
            continue
    casedir = datadir + '/' + c

    # study dirs. for now, only 1 study is segmented
    res = dbox.files_list_folder(casedir)

    for rdir in res.entries:
        
        sdir = casedir + '/' + rdir.name
        res2 = dbox.files_list_folder(sdir)
        rlist = [x.name for x in res2.entries]
        # mask_T is not present if no tumor
        if set(flist[1:]).issubset(set(rlist)):

            # for individual files
            for f in flist:
                try:
                    md,res3 = dbox.files_download(sdir + '/' + f)
                except dropbox.exceptions.HttpError as e:
                    print('HTTP error',e)
                    print('skipping {}'.format(f))
                    continue
                except ConnectionError as e:
                    print('HTTP error',e)
                    print('attempt to re-connect')
                    # this isn't global scope though
                    dbox = dropbox.Dropbox(token)
                    md,res = dbox.files_download(sdir + '/' + f)
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
                    # can't run a process by os.system on the non-boot drive in windows?
                    if 'mask' in f and os.name == 'posix':
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
