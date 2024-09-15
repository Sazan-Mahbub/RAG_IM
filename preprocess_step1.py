from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data_dir = './data_v2'


allowed_icd_codes = {'02HV33Z', '5A1D70Z', '3E0G76Z', '5A1D60Z', '0DJ08ZZ', '0BH17EZ', '0W9G3ZZ', 'B211YZZ', '02H633Z', '5A1945Z', '0W9G3ZX', '0DJD8ZZ', '5A1955Z', '009U3ZX', '3E0436Z', '5A1D00Z', '3E04305', '0DB68ZX', '5A1935Z', '10E0XZZ', '4A023N6', '4A023N7', '3E0H76Z', '0W9930Z', 'B548ZZA', 'GZB0ZZZ', '5A2204Z', '5A1221Z', '0DB98ZX', 'GZB2ZZZ', '0JH63XZ', '0DH63UZ', '027034Z', '0W9B30Z', '3E1M39Z', '0F798DZ', '0W3P8ZZ', 'B2111ZZ', '0BJ08ZZ', '07DR3ZX', '10D00Z1', '0T25X0Z', '0CJS8ZZ', 'BF10YZZ', '0FC98ZZ', '0JBR0ZZ', '0DBN8ZX', '30233S1', '3E03305', '0W9G30Z'}

HOSP = pd.read_csv(filepath_or_buffer=f'{data_dir}/HOSP.procedures_icd.csv')
HOSP = HOSP[HOSP['icd_code'].isin(allowed_icd_codes)]
subject_id_set_hosp = HOSP['subject_id'].unique().tolist()
subject_id_set_hosp = set(subject_id_set_hosp)


diagnosis = pd.read_csv(filepath_or_buffer=f'{data_dir}/ED.diagnosis.csv')
diagnosis = diagnosis[diagnosis['icd_version'] == 10]
subject_id_set = diagnosis['subject_id'].unique().tolist()
subject_id_set = set(subject_id_set)


filtered_subjects_set = subject_id_set.intersection(subject_id_set_hosp)

filtered_labevents = open(f'{data_dir}/filtered_labevents__all_icd_code.csv', 'w')

with open(f'{data_dir}/labevents.csv') as f:
    cnt = 0
    cnt_success = 0
    for line in f:
        # print('>>', cnt)
        # print(line.split(','))
        sub_id = str(line.split(',')[1])
        # print(sub_id, type(sub_id))
        
        if sub_id == 'subject_id':
            filtered_labevents.write(line)
            continue
        else:
            sub_id = int(sub_id)
            
        if sub_id in filtered_subjects_set: 
            filtered_labevents.write(line)
            cnt_success += 1
            # print('Success')
        else:
            # print(cnt, ' [WARNING] :', sub_id, type(sub_id), 'not in labevents.csv')
            # raise
            pass
        cnt += 1
        if cnt % 100_000_000 == 0:
            print('Current total count:', cnt, 'Current success count:', cnt_success)

filtered_labevents.close()

print('Total:', cnt, '\tTotal success:', cnt_success)
print('Done')


# filtered_labevents = open('filtered_labevents__all_icd_code_split1.csv', 'w')

# with open('labevents.csv') as f:
#     cnt = 0
#     cnt_success = 0
#     for line in f:
#         # print('>>', cnt)
#         # print(line.split(','))
#         sub_id = str(line.split(',')[1])
#         # print(sub_id, type(sub_id))
        
#         if sub_id == 'subject_id':
#             filtered_labevents.write(line)
#             continue
#         else:
#             sub_id = int(sub_id)
            
#         if sub_id in filtered_subjects_set: 
#             filtered_labevents.write(line)
#             cnt_success += 1
#             if cnt_success == int(37489622 / 2):
#                 filtered_labevents.close()
#                 filtered_labevents = open('filtered_labevents__all_icd_code_split2.csv', 'w')
#             # print('Success')
#         else:
#             # print(cnt, ' [WARNING] :', sub_id, type(sub_id), 'not in labevents.csv')
#             # raise
#             pass
#         cnt += 1
#         if cnt % 100_000_000 == 0:
#             print('Current total count:', cnt, 'Current success count:', cnt_success)

# filtered_labevents.close()

# print('Total:', cnt, '\tTotal success:', cnt_success)
# print('Done')



